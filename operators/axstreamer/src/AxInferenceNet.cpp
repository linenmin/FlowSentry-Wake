// Copyright Axelera AI, 2025
#include "AxInferenceNet.hpp"
#include "AxDataInterface.h"
#include "AxInference.hpp"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxStreamerUtils.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>


namespace Ax
{

struct Frame : public CompletedFrame {
  SharedBatchBufferView op_input;
  bool skip_inference = false;
  int stream_id;
  int subframe_index = 0;
  int number_of_subframes = 1;
  time_point timestamp;
  time_point latency_start;
};

using Buffers = std::list<ManagedDataInterface>;

class Operator
{
  public:
  //  Note: input will be consumed by the operator
  //  It might be returned in the output if the operator is in-place
  virtual SharedBatchBufferView execute(const AxVideoInterface &video,
      SharedBatchBufferView input, unsigned int subframe_index,
      unsigned int number_of_subframes, MetaMap &meta_map)
      = 0;

  virtual std::string name() const = 0;

  virtual AxDataInterface allocate_output(const AxDataInterface &input) = 0;

  virtual void set_allocator(int batch_size, std::unique_ptr<DataInterfaceAllocator> &&) = 0;

  virtual bool supports_crop() const = 0;

  virtual void downstream_supports_crop(bool) = 0;

  virtual bool consumes_frames() const = 0;

  virtual ~Operator() = default;
};

class OperatorList
{
  public:
  explicit OperatorList(Logger &logger, LatencyCallback log_latency)
      : logger(logger), log_latency(log_latency)
  {
  }

  void add_operator(std::string libname, std::string options,
      std::string mode = "none", std::string batch_size = "1");

  AxDataInterface compile(const AxDataInterface &input, DataInterfaceAllocator &allocator,
      std::function<void(std::unique_ptr<Frame>)> release_frame);

  void set_allocator(int batch_size, std::unique_ptr<DataInterfaceAllocator> &&allocator)
  {
    operators.back()->set_allocator(batch_size, std::move(allocator));
  }

  BlockingQueue<std::unique_ptr<Ax::Frame>> &input_queue()
  {
    return links.front();
  }

  BlockingQueue<std::unique_ptr<Ax::Frame>> &output_queue()
  {
    return links.back();
  }

  void stop()
  {
    for (auto &q : links) {
      q.stop();
    }
  }

  void initialise()
  {
    auto first = operators.begin();
    auto last = operators.end();
    if (first == last)
      return;

    for (auto prev = first; ++first != last; prev = first) {
      (*prev)->downstream_supports_crop((*first)->supports_crop());
    }
  }

  private:
  void log_throughput(const std::string &op, time_point throughput_start,
      time_point latency_start);

  void operator_thread(int which, std::function<void(std::unique_ptr<Ax::Frame>)> release_frame);

  Logger &logger;
  LatencyCallback log_latency;
  std::vector<std::unique_ptr<Operator>> operators;
  struct OpCallParams {
    Operator *op;
  };
  std::vector<OpCallParams> params;
  std::vector<std::jthread> threads;
  std::vector<BlockingQueue<std::unique_ptr<Ax::Frame>>> links;
};

class AxInferenceNet : public InferenceNet
{
  public:
  AxInferenceNet(const InferenceNetProperties &properties, Ax::Logger &logger,
      InferenceDoneCallback done_callback, LatencyCallback latency_callback);

  void push_new_frame(std::shared_ptr<void> &&buffer,
      const AxVideoInterface &video, MetaMap &axmetamap, int stream_id) override;
  void stop() override;
  void end_of_input() override;
  void cascade_frame(CompletedFrame &frame) override;

  private:
  void release_frame(std::unique_ptr<Frame> frame);
  void init_frame(Ax::Frame &frame);
  std::unique_ptr<Frame> build_eos_frame(
      Frame *last_frame, int batch, std::shared_ptr<BatchedBuffer> batched);
  bool unbatch(std::queue<std::unique_ptr<Frame>> &pending_frames,
      std::shared_ptr<BatchedBuffer> out, int batch_size, int current_batch,
      const ManagedDataInterface &output);
  std::unique_ptr<Frame> next_frame(bool at_eos, const SharedBatchBufferView &last_good_input);

  struct Stream {
    std::atomic_uint64_t frame_id{ 0 };
    std::chrono::microseconds latency;
    int count = 0;
  };

  void inference_thread(const int batch_size);
  std::unique_ptr<Frame> new_frame();
  void initialise_pipeline(const AxVideoInterface &video);
  void update_stream_latency(int which, std::chrono::microseconds latency);
  void update_frame_latency(Frame &frame, const char *label);
  void log_latency(const std::string &op, std::chrono::high_resolution_clock::time_point start);
  void finalize_thread();

  const InferenceNetProperties properties;
  Logger &logger;
  std::unique_ptr<Inference> inference;
  OperatorList pre_ops;
  OperatorList post_ops;

  std::vector<std::jthread> threads;
  std::jthread push_thread;
  std::vector<std::unique_ptr<Frame>> frame_pool;
  std::mutex frame_pool_mutex;

  // note the order of destruction is really important here. will need to tidy
  // this up but the queues must be destroyed before the pools
  std::unique_ptr<DataInterfaceAllocator> inf_input_allocator;
  std::unique_ptr<DataInterfaceAllocator> inf_output_allocator;
  std::unique_ptr<DataInterfaceAllocator> allocator;
  std::unique_ptr<BatchedBufferPool> inf_input_pool;
  std::unique_ptr<BatchedBufferPool> inf_output_pool;
  std::unordered_map<int, Stream> streams;
  std::once_flag compile_once_flag;
  InferenceDoneCallback done_callback;
  LatencyCallback latency_callback;
  int num_to_flush = 0;
};

std::unordered_map<std::string, int> plugin_names_count{};

std::string
make_plugin_name(std::string n)
{
  auto name = std::move(n);
  if (name.ends_with(".so")) {
    name = name.substr(0, name.size() - 3);
  }
  auto count = plugin_names_count[name]++;
  return name + "_" + std::to_string(count);
}

const auto empty_allowed = StringSet{};

template <typename PluginType> struct AxPlugin {

  AxPlugin(Ax::Logger &logger, Ax::SharedLib &&shared_, std::string options,
      std::string mode = "none")
      : logger(logger), shared(std::move(shared_)),
        name_(make_plugin_name(shared.libname()))
  {
    Ax::load_v1_plugin(shared, fns);
    const auto &allowed = fns.allowed_properties ? fns.allowed_properties() : empty_allowed;
    auto opts = Ax::parse_and_validate_plugin_options(logger, options, allowed);
    if (fns.init_and_set_static_properties) {
      subplugin_data = fns.init_and_set_static_properties(opts, logger);
    }
    if (fns.set_dynamic_properties) {
      fns.set_dynamic_properties(opts, subplugin_data.get(), logger);
    }
  }

  std::string name() const
  {
    return name_;
  }

  ~AxPlugin() = default;

  Ax::Logger &logger;
  Ax::SharedLib shared;
  PluginType fns;
  std::shared_ptr<void> subplugin_data;
  std::string name_;
};

class TransformOp : public Ax::Operator
{
  public:
  TransformOp(Ax::Logger &logger, Ax::SharedLib &&shared_, std::string libname,
      std::string options, std::string mode, int batch_size,
      std::unique_ptr<Ax::DataInterfaceAllocator> &&alloc)
      : plugin(logger, std::move(shared_), options, mode), logger(logger),
        allocator(std::move(alloc)), pool{ std::make_unique<Ax::BatchedBufferPool>(
                                         batch_size, AxDataInterface{}, *allocator) },
        batch{ batch_size }
  {
  }

  void remove_cropinfo(AxDataInterface &out)
  {
    if (auto *video = std::get_if<AxVideoInterface>(&out)) {
      video->info.stride
          = video->info.width * AxVideoFormatNumChannels(video->info.format);
      video->strides = { size_t(video->info.stride) };
      video->info.cropped = false;
      video->info.x_offset = 0;
      video->info.y_offset = 0;
    }
  }

  Ax::SharedBatchBufferView batch_output(std::shared_ptr<Ax::BatchedBuffer> output_buffer)
  {
    if (++current_batch != batch) {
      out_buf = std::move(output_buffer);
      return get_shared_view_of_batch_buffer(out_buf, 0);
    }
    current_batch = 0;
    out_buf.reset();
    return get_shared_view_of_batch_buffer(output_buffer, 0);
  }


  Ax::SharedBatchBufferView finish_previous_transform(Ax::SharedBatchBufferView in_buffer,
      Ax::SharedBatchBufferView out_buffer, AsyncCompleter complete_last_xform)
  {
    if (dbl_buffer.inbuffer) {
      //  We have a pending buffer
      std::swap(dbl_buffer.complete_last_xform, complete_last_xform);
      std::swap(dbl_buffer.inbuffer, in_buffer);
      std::swap(dbl_buffer.outbuffer, out_buffer);
      complete_last_xform();
      return batch_output(out_buffer.underlying());
    } else {
      //  No pending buffer, just set up next buffer
      dbl_buffer.complete_last_xform = std::move(complete_last_xform);
      dbl_buffer.inbuffer = std::move(in_buffer);
      dbl_buffer.outbuffer = std::move(out_buffer);
      return {};
    }
  }

  //  Should only be called if passthrough is possible
  AxDataInterface assign_input_buffers_to_output_buffers(
      const AxDataInterface &input, const AxDataInterface &out)
  {
    auto output = out;
    if (std::holds_alternative<AxVideoInterface>(input)) {
      auto *data = std::get<AxVideoInterface>(input).data;
      if (std::holds_alternative<AxVideoInterface>(output)) {
        auto &out_video = std::get<AxVideoInterface>(output);
        out_video.data = data;
      } else if (std::holds_alternative<AxTensorsInterface>(output)) {
        auto &out_tensors = std::get<AxTensorsInterface>(output);
        out_tensors[0].data = data;
      }
      return output;
    }
    if (std::holds_alternative<AxVideoInterface>(output)) {
      auto &out_video = std::get<AxVideoInterface>(output);
      out_video.data = std::get<AxTensorsInterface>(input)[0].data;
    } else if (std::holds_alternative<AxTensorsInterface>(output)) {
      auto &out_tensors = std::get<AxTensorsInterface>(output);
      auto &in_tensors = std::get<AxTensorsInterface>(input);
      for (size_t i = 0; i < in_tensors.size(); ++i) {
        out_tensors[i].data = in_tensors[i].data;
      }
    }
    return output;
  }

  Ax::SharedBatchBufferView execute(const AxVideoInterface &video,
      Ax::SharedBatchBufferView input, unsigned int subframe_index,
      unsigned int number_of_subframes, Ax::MetaMap &meta_map) override
  {
    (void) video;
    auto use_double_buffer = Ax::get_env("AXELERA_USE_CL_DOUBLE_BUFFER", "1");
    if (!input) {
      return dbl_buffer.inbuffer ? finish_previous_transform({}, {}, [] {}) : input;
    }
    auto out = set_output_interface(*input, plugin.subplugin_data.get(), logger);

    if (plugin.fns.set_output_interface_from_meta) {
      out = plugin.fns.set_output_interface_from_meta(*input,
          plugin.subplugin_data.get(), subframe_index, number_of_subframes,
          meta_map, logger);
      if (std::holds_alternative<AxVideoInterface>(out)) {
        auto &out_video = std::get<AxVideoInterface>(out);
        if (downstream_supports_cropmeta && out_video.info.cropped) {
          auto in = input.underlying();
          in->set_iface(out);
          //  We either always add crop metadata or we don't
          //  In which case we do not need to worry about double buffering
          return input;
        }
      }
      remove_cropinfo(out);
    }

    if (batch == 1 && plugin.fns.can_passthrough
        && plugin.fns.can_passthrough(*input, out, plugin.subplugin_data.get(), logger)) {
      //  If we get here we can passthrough the input to the output with just moidified caps
      out = assign_input_buffers_to_output_buffers(*input, out);
      auto in = input.underlying();
      in->set_iface(out);
      auto output = input;
      return dbl_buffer.inbuffer ? finish_previous_transform(input, output, [] {}) : output;
    }

    //  Here we check if output comes from meta and if so get the output
    //  interface Otherwise it comes from set_output_interface
    if (std::holds_alternative<AxVideoInterface>(out)) {
      auto &out_video = std::get<AxVideoInterface>(out);
      if (downstream_supports_cropmeta && out_video.info.cropped) {
        auto in = input.underlying();
        in->set_iface(out);
        //  We either always add crop metadata or we don't
        //  In which case we do not need to worry about double buffering
        return input;
      }
      //  Remove cropping metadata if we need to physically crop here.
      remove_cropinfo(out);
    }
    auto output_buffer = out_buf ? out_buf : pool->new_batched_buffer(out, true);
    auto output = get_shared_view_of_batch_buffer(output_buffer, current_batch);

    if (plugin.fns.transform_async) {
      auto complete = plugin.fns.transform_async(*input, *output,
          plugin.subplugin_data.get(), subframe_index, number_of_subframes,
          meta_map, logger);
      if (use_double_buffer == "1") {
        return finish_previous_transform(
            input, get_shared_view_of_batch_buffer(output_buffer, 0), complete);
      }
      complete();
      return batch_output(std::move(output_buffer));
    }

    plugin.fns.transform(*input, *output, plugin.subplugin_data.get(),
        subframe_index, number_of_subframes, meta_map, logger);
    return batch_output(std::move(output_buffer));
  }

  AxDataInterface allocate_output(const AxDataInterface &input) override
  {
    auto output = set_output_interface(input, plugin.subplugin_data.get(), logger);
    if (batch != 1) {
      if (!std::holds_alternative<AxTensorsInterface>(output)) {
        throw std::runtime_error("Batched output must be tensor");
      }
      auto &tensors = std::get<AxTensorsInterface>(output);
      tensors[0].sizes[0] = batch;
    }
    return output;
  }

  void set_allocator(int batch_size, std::unique_ptr<Ax::DataInterfaceAllocator> &&alloc) override
  {
    allocator = std::move(alloc);
    pool = std::make_unique<Ax::BatchedBufferPool>(batch_size, AxDataInterface{}, *allocator);
  }

  std::string name() const override
  {
    return plugin.name();
  }

  bool supports_crop() const override
  {
    return plugin.fns.handles_crop_meta && plugin.fns.handles_crop_meta();
  }

  void downstream_supports_crop(bool supports) override
  {
    downstream_supports_cropmeta = supports;
  }

  bool consumes_frames() const override
  {
    return false;
  }

  private:
  struct double_buffer_details {
    Ax::SharedBatchBufferView inbuffer{};
    Ax::SharedBatchBufferView outbuffer{};
    Ax::AsyncCompleter complete_last_xform{};
  };

  AxDataInterface set_output_interface(
      const AxDataInterface &interface, const void *prop, Ax::Logger &logger)
  {
    return plugin.fns.set_output_interface ?
               plugin.fns.set_output_interface(interface, prop, logger) :
               interface;
  }

  AxPlugin<Ax::V1Plugin::Transform> plugin;
  Ax::Logger &logger;
  std::unique_ptr<Ax::DataInterfaceAllocator> allocator;
  std::unique_ptr<Ax::BatchedBufferPool> pool;
  bool downstream_supports_cropmeta = false;
  double_buffer_details dbl_buffer;
  std::shared_ptr<Ax::BatchedBuffer> out_buf;
  int batch = 1;
  int current_batch = 0;
};


class InplaceOp : public Ax::Operator
{
  public:
  InplaceOp(Ax::Logger &logger, Ax::SharedLib &&shared, std::string libname,
      std::string options, std::string mode)
      : plugin(logger, std::move(shared), options, mode), logger(logger)
  {
  }

  Ax::SharedBatchBufferView execute(const AxVideoInterface &video,
      Ax::SharedBatchBufferView input, unsigned int subframe_index,
      unsigned int number_of_subframes, Ax::MetaMap &meta_map) override
  {
    (void) video; // not used
    if (!input) {
      return input;
    }
    plugin.fns.inplace(*input, plugin.subplugin_data.get(), subframe_index,
        number_of_subframes, meta_map, logger);
    return input;
  }


  AxDataInterface allocate_output(const AxDataInterface &input) override
  {
    return input;
  }

  std::string name() const override
  {
    return plugin.name();
  }

  void set_allocator(int /*batch_size*/, std::unique_ptr<Ax::DataInterfaceAllocator> &&) override
  {
  }

  bool supports_crop() const override
  {
    return false;
  }

  void downstream_supports_crop(bool) override
  {
  }

  bool consumes_frames() const override
  {
    return false;
  }

  private:
  AxPlugin<Ax::V1Plugin::InPlace> plugin;
  Ax::Logger &logger;
};

class DecodeOp : public Ax::Operator
{
  public:
  DecodeOp(Ax::Logger &logger, Ax::SharedLib &&shared, std::string libname,
      std::string options, std::string mode)
      : plugin(logger, std::move(shared), options, mode), logger(logger),
        allocator(std::make_unique<Ax::NullDataInterfaceAllocator>()), pool{
          std::make_unique<Ax::BatchedBufferPool>(1, AxDataInterface{}, *allocator)
        }
  {
  }

  Ax::SharedBatchBufferView execute(const AxVideoInterface &video,
      Ax::SharedBatchBufferView input, unsigned int subframe_index,
      unsigned int number_of_subframes, Ax::MetaMap &meta_map) override
  {
    if (!input) {
      return input;
    }
    auto &tensor = std::get<AxTensorsInterface>(*input);
    auto out = pool->new_batched_buffer(video, false);
    if (number_of_subframes != 0) {
      plugin.fns.decode_to_meta(tensor, plugin.subplugin_data.get(),
          subframe_index, number_of_subframes, meta_map, video, logger);
    }
    return number_of_subframes == 0 || number_of_subframes == subframe_index + 1 ?
               get_shared_view_of_batch_buffer(out, 0) :
               Ax::SharedBatchBufferView{};
  }

  AxDataInterface allocate_output(const AxDataInterface &input) override
  {
    return {};
  }
  std::string name() const override
  {
    return plugin.name();
  }

  void set_allocator(int /*batch_size*/, std::unique_ptr<Ax::DataInterfaceAllocator> &&) override
  {
  }

  bool supports_crop() const override
  {
    return false;
  }

  void downstream_supports_crop(bool) override
  {
  }

  bool consumes_frames() const override
  {
    return true;
  }

  private:
  AxPlugin<Ax::V1Plugin::Decoder> plugin;
  Ax::Logger &logger;
  std::unique_ptr<Ax::DataInterfaceAllocator> allocator;
  std::unique_ptr<Ax::BatchedBufferPool> pool;
};

static AxDataInterface
strip_pointers(AxDataInterface in)
{
  // return a copy of the output interface without the ->data pointers, as
  // this represents the output 'shape', not the actual data interface
  if (auto *video = std::get_if<AxVideoInterface>(&in)) {
    video->data = nullptr;
  } else if (auto *tensors = std::get_if<AxTensorsInterface>(&in)) {
    for (auto &tensor : *tensors) {
      tensor.data = nullptr;
    }
  }
  return in;
}

using time_point = Ax::time_point;

auto
as_duration(time_point start, time_point end)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

auto
as_duration_ns(time_point start, time_point end)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

auto
duration_since(time_point start)
{
  return as_duration(start, std::chrono::high_resolution_clock::now());
}

auto
duration_since_ns(time_point start)
{
  return as_duration_ns(start, std::chrono::high_resolution_clock::now());
}

void
Ax::OperatorList::operator_thread(
    int which, std::function<void(std::unique_ptr<Ax::Frame>)> release_frame)
{
  auto &input_queue = links[which];
  auto &output_queue = links[which + 1];
  auto &op = *operators[which];
  auto previous_frame = std::unique_ptr<Ax::Frame>{};
  while (true) {
    auto frame = input_queue.wait_one();
    if (!frame) {
      return;
    }
    auto now = std::chrono::high_resolution_clock::now();
    frame->latency_start = now;
    static Ax::MetaMap dummy_meta_map{};
    auto &meta_map = frame->meta_map ? *frame->meta_map : dummy_meta_map;
    if (frame->end_of_input) {
      //  Here we just flush any pending frames and then send on the
      //  EOS/Gap frame
      auto out = op.execute({}, {}, 0, 1, meta_map);
      log_throughput(op.name(), now, frame->latency_start);
      if (out) {
        previous_frame->op_input = std::move(out);
        output_queue.push({ std::move(previous_frame) });
      }
      frame->op_input = {};
      output_queue.push(std::move(frame));
    } else {
      const auto &video = frame->video;
      auto subframe_index = frame->subframe_index;
      auto number_of_subframes = frame->number_of_subframes;
      auto out = op.execute(video, std::move(frame->op_input), subframe_index,
          number_of_subframes, meta_map);
      log_throughput(op.name(), now, frame->latency_start);
      if (out) {
        if (op.consumes_frames()) {
          //  After the decoder we can reset subframes
          frame->subframe_index = 0;
          frame->number_of_subframes = 1;
        }
        if (previous_frame) {
          std::swap(frame, previous_frame);
        }
        frame->op_input = std::move(out);
        output_queue.push({ std::move(frame) });
      } else if (op.consumes_frames()) {
        release_frame(std::move(frame));
      } else {
        previous_frame = std::move(frame);
      }
    }
  }
}

void
OperatorList::add_operator(std::string libname, std::string options,
    std::string mode /*= "none"*/, std::string batch_size)
{
  auto batch_size_int = batch_size.empty() ? 1 : std::stoi(batch_size);
  Ax::SharedLib shared(logger, libname);
  if (shared.has_symbol("transform") || shared.has_symbol("transform_async")) {
    operators.push_back(std::make_unique<TransformOp>(logger, std::move(shared),
        libname, options, mode, batch_size_int, create_heap_allocator()));
  } else if (shared.has_symbol("inplace")) {
    operators.push_back(std::make_unique<InplaceOp>(
        logger, std::move(shared), libname, options, mode));
  } else if (shared.has_symbol("decode_to_meta")) {
    operators.push_back(std::make_unique<DecodeOp>(
        logger, std::move(shared), libname, options, mode));
  } else {
    throw std::runtime_error("Unknown module " + libname);
  }
}


AxDataInterface
Ax::OperatorList::compile(const AxDataInterface &input, Ax::DataInterfaceAllocator &allocator,
    std::function<void(std::unique_ptr<Ax::Frame>)> release_frame)
{
  auto size = operators.size();
  std::vector<BlockingQueue<std::unique_ptr<Ax::Frame>>> queues(size + 1);
  links = std::move(queues);
  AxDataInterface in = input;
  for (int i = 0; i != operators.size(); ++i) {
    auto out = operators[i]->allocate_output(in);
    threads.emplace_back(std::jthread(&OperatorList::operator_thread, this, i, release_frame));
    in = out;
  }
  return strip_pointers(in);
}

void
Ax::OperatorList::log_throughput(
    const std::string &op, time_point throughput_start, time_point latency_start)
{
  log_latency(op, duration_since_ns(throughput_start).count(),
      duration_since_ns(latency_start).count());
}

AxInferenceNet::AxInferenceNet(const Ax::InferenceNetProperties &properties,
    Ax::Logger &logger, InferenceDoneCallback done_callback, LatencyCallback latency_callback)
    : properties(properties), logger(logger),
      inference(create_inference(logger, properties)),
      pre_ops(logger, latency_callback), post_ops(logger, latency_callback),
      allocator(create_heap_allocator()), done_callback(done_callback),
      latency_callback(latency_callback)
{
  logger(AX_INFO) << "InferenceNet created" << std::endl;
}

void
AxInferenceNet::update_frame_latency(Ax::Frame &frame, const char *label)
{
  if (!frame.end_of_input) {
    auto now = std::chrono::high_resolution_clock::now();
    auto start = std::exchange(frame.latency_start, now);
    log_latency(label, start);
  }
}

bool
AxInferenceNet::unbatch(std::queue<std::unique_ptr<Frame>> &pending_frames,
    std::shared_ptr<Ax::BatchedBuffer> out, int batch_size, int current_batch,
    const Ax::ManagedDataInterface &output)
{
  auto eos = false;
  //  If we are here and have not collected a full batch we are at end of
  //  stream and need to forward whateer frames we have (and no more)
  auto num_frames = current_batch != 0 ? current_batch : batch_size;
  for (int n = 0; n != num_frames && !eos;) {
    auto out_frame = Ax::pop_queue(pending_frames);
    out_frame->op_input = get_shared_view_of_batch_buffer(out, n++);
    eos = out_frame->end_of_input;
    update_frame_latency(*out_frame, "Inference latency");
    post_ops.input_queue().push(std::move(out_frame));
  }
  //  Push any following gap frames
  return eos;
}

struct params {
  std::shared_ptr<Ax::BatchedBuffer> input;
  std::shared_ptr<Ax::BatchedBuffer> output;
};

// This delivers frames from the queue whilst we are not at EOS
//  Once at EOS it will deliver enough frames to flush any pending buffers out
//  of the end Of the queue. It will then block until the queue terminates.
//
std::unique_ptr<Ax::Frame>
AxInferenceNet::next_frame(bool at_eos, const Ax::SharedBatchBufferView &last_good_input)
{
  if (at_eos && num_to_flush != 0) {
    --num_to_flush;
    auto fr = new_frame();
    fr->op_input = last_good_input;
    fr->end_of_input = num_to_flush == 0;
    fr->frame_id = -2;
    fr->timestamp = std::chrono::high_resolution_clock::now();
    fr->latency_start = std::chrono::high_resolution_clock::now();
    return fr;
  }
  auto frame = pre_ops.output_queue().wait_one();
  return frame;
}

void
AxInferenceNet::log_latency(
    const std::string &op, std::chrono::high_resolution_clock::time_point start)
{
  auto now = std::chrono::high_resolution_clock::now();
  auto latency = as_duration_ns(start, now);
  latency_callback(op, latency.count(), latency.count());
}

void
AxInferenceNet::inference_thread(const int batch_size)
{
  std::queue<std::unique_ptr<Frame>> pending_frames;
  std::queue<params> pending_params;

  auto num_to_drop = output_drop(properties);
  auto pre_fill = pipeline_pre_fill(properties);
  bool at_eos = false;
  auto this_batched_input = inf_input_pool->new_batched_buffer(false);
  auto last_good_input = get_shared_view_of_batch_buffer(this_batched_input, 0);
  int current_batch = 0;
  while (true) {
    auto current = std::chrono::high_resolution_clock::now();
    auto frame = next_frame(at_eos, last_good_input);
    if (!frame) {
      return;
    }
    if (frame->end_of_input) {
      if (!at_eos) {
        at_eos = true;
        num_to_flush
            = 1 + batch_size * (output_drop(properties) + pipeline_pre_fill(properties));
        log_latency("inference", current);
        continue;
      }
      if (current_batch != 0) {
        //  We do not have a complete batch so we need to seend wahtever frames are leftover
        auto [op_input, inf_output] = Ax::pop_queue(pending_params);
        inf_output->map();
        inference->collect({});
        auto eos = unbatch(pending_frames, inf_output, batch_size,
            current_batch, inf_output->get_batched(false));
      }
      post_ops.input_queue().push(std::move(frame));
      log_latency("inference", current);
      continue;
    }
    if (current_batch == 0) {
      //  We use the inpput from the first frane of the batch
      last_good_input = frame->op_input;
    }
    auto batched_input = last_good_input.underlying();
    pending_frames.push(std::move(frame));
    if (++current_batch != batched_input->batch_size()) {
      log_latency("inference", current);
      continue;
    }
    current_batch = 0;
    auto batched_output = inf_output_pool->new_batched_buffer(false);
    const auto &input = batched_input->get_batched(properties.dmabuf_inputs);
    const auto &output = batched_output->get_batched(properties.dmabuf_outputs);
    pending_params.push({ batched_input, batched_output });
    inference->dispatch(input.buffers(), input.fds(), output.buffers(), output.fds());
    if (pre_fill) {
      --pre_fill;
      log_latency("inference", current);
      continue;
    }

    auto [op_input, inf_output] = Ax::pop_queue(pending_params);
    if (num_to_drop) {
      --num_to_drop;
      inference->collect(inf_output->get_batched(false).buffers());
      log_latency("inference", current);
      continue;
    }
    inf_output->map();
    inference->collect({});
    log_latency("inference", current);
    //  This can potentially block, so we do not want to include it in the latency
    auto eos = unbatch(pending_frames, inf_output, batch_size, current_batch,
        inf_output->get_batched(false));
    if (eos) {
      Ax::clear_queue(pending_frames);
      Ax::clear_queue(pending_params);
      break;
    }
  }
}

void
AxInferenceNet::update_stream_latency(int which, std::chrono::microseconds latency)
{
  streams[which].latency += latency;
  streams[which].count += 1;
}

void
AxInferenceNet::finalize_thread()
{
  while (true) {
    auto frame = post_ops.output_queue().wait_one();
    if (!frame) {
      return;
    }
    frame->subframe_index = 0;
    frame->number_of_subframes = 1;
    update_frame_latency(*frame, "Postprocessing latency");
    if (!frame->end_of_input) {
      log_latency("Total latency", frame->timestamp);
    }
    done_callback(*frame);
    update_stream_latency(frame->stream_id, duration_since(frame->timestamp));
    release_frame(std::move(frame));
  }
}

void
AxInferenceNet::stop()
{
  pre_ops.stop();
  post_ops.stop();
  threads.clear();
}

std::unique_ptr<Ax::Frame>
AxInferenceNet::new_frame()
{
  std::unique_lock<std::mutex> lock(frame_pool_mutex);
  if (frame_pool.empty()) {
    return std::make_unique<Frame>();
  }
  auto frame = std::move(frame_pool.back());
  frame_pool.pop_back();
  return frame;
}

void
AxInferenceNet::release_frame(std::unique_ptr<Frame> frame)
{
  frame->stream_id = 0;
  frame->frame_id = 0;
  frame->buffer_handle.reset();
  frame->video.data = nullptr;
  frame->meta_map = nullptr;
  frame->op_input.reset();
  frame->end_of_input = false;
  std::unique_lock<std::mutex> lock(frame_pool_mutex);
  frame_pool.push_back(std::move(frame));
}

void
AxInferenceNet::init_frame(Ax::Frame &frame)
{
  AxDataInterface in{ frame.video };
  auto null_alloc = Ax::NullDataInterfaceAllocator();
  auto input = std::make_shared<Ax::BatchedBuffer>(1, in, null_alloc);
  frame.op_input = get_shared_view_of_batch_buffer(input, 0);
}

static size_t
get_number_of_subframes(const MetaMap &axmetamap, const std::string &meta_to_distribute)
{
  if (!meta_to_distribute.empty()) {
    auto meta = axmetamap.find(meta_to_distribute);
    if (meta == axmetamap.end()) {
      throw std::runtime_error("Meta " + meta_to_distribute + " not found, cannot distribute");
    }
    return meta->second->get_number_of_subframes();
  }
  return 1;
}

void
AxInferenceNet::push_new_frame(std::shared_ptr<void> &&buffer_handle,
    const AxVideoInterface &video, MetaMap &axmetamap, int stream_id)
{
  auto &stream = streams[stream_id];
  std::call_once(compile_once_flag, [this, video] { initialise_pipeline(video); });

  auto frame = new_frame();
  frame->buffer_handle = buffer_handle;
  frame->video = video;
  frame->meta_map = &axmetamap;
  frame->stream_id = stream_id;
  auto frame_id = stream.frame_id++;
  frame->frame_id = frame_id;
  if (properties.skip_stride > 1 && properties.skip_count > 0) {
    const auto reverse_index_in_slice
        = properties.skip_stride - (frame->frame_id % properties.skip_stride) - 1;
    frame->skip_inference = reverse_index_in_slice < properties.skip_count;
  }
  frame->end_of_input = false;
  frame->timestamp = std::chrono::high_resolution_clock::now();
  frame->latency_start = std::chrono::high_resolution_clock::now();
  auto &preq = pre_ops.input_queue();
  const auto num = get_number_of_subframes(axmetamap, properties.meta);
  //  It is one or more subframes
  frame->subframe_index = 0;
  frame->number_of_subframes = num;
  init_frame(*frame);
  preq.push(std::move(frame));
  for (int n = 1; n < num; ++n) {
    auto subframe = new_frame();
    subframe->buffer_handle = buffer_handle;
    subframe->video = video;
    subframe->meta_map = &axmetamap;
    subframe->stream_id = stream_id;
    subframe->frame_id = frame_id;
    subframe->subframe_index = n;
    subframe->number_of_subframes = num;
    subframe->end_of_input = false;
    subframe->timestamp = std::chrono::high_resolution_clock::now();
    subframe->latency_start = std::chrono::high_resolution_clock::now();
    init_frame(*subframe);
    preq.push(std::move(subframe));
  }
}

void
AxInferenceNet::cascade_frame(CompletedFrame &frame)
{
  if (frame.end_of_input) {
    end_of_input();
  } else {
    push_new_frame(std::move(frame.buffer_handle), frame.video, *frame.meta_map,
        frame.stream_id);
  }
}

void
AxInferenceNet::initialise_pipeline(const AxVideoInterface &video)
{
  auto compile_list = [this](const decltype(properties.preproc) &props, OperatorList &list) {
    for (int n = 0; n != MAX_OPERATORS && !props[n].lib.empty(); ++n) {
      logger(AX_INFO)
          << "Adding operator: " << props[n].lib << "(" << props[n].options
          << ", " << props[n].mode << ", " << props[n].batch << ")" << std::endl;
      list.add_operator(props[n].lib, props[n].options, props[n].mode, props[n].batch);
    }
    list.initialise();
  };

  compile_list(properties.preproc, pre_ops);
  compile_list(properties.postproc, post_ops);
  ManagedDataInterfaces buffers;
  auto releaser = std::function<void(std::unique_ptr<Ax::Frame>)>(
      [this](std::unique_ptr<Ax::Frame> frame) {
        return release_frame(std::move(frame));
      });
  auto inf_input_template = pre_ops.compile(video, *allocator, releaser);
  const auto exp_model_input
      = Ax::to_string(AxDataInterface(inference->input_shapes()));
  const auto got_model_input = Ax::to_string(inf_input_template);
  const auto batch_size = inference->batch_size();
  if (exp_model_input != got_model_input) {
    throw std::runtime_error("Expected model input=" + exp_model_input
                             + " but got input=" + got_model_input);
  }
  auto inf_output_template = inference->output_shapes();
  post_ops.compile(Ax::batch_view(inf_output_template, 0), *allocator, releaser);

  inf_input_allocator = properties.dmabuf_inputs ? create_dma_buf_allocator() :
                                                   create_heap_allocator();
  inf_output_allocator = properties.dmabuf_outputs ? create_dma_buf_allocator() :
                                                     create_heap_allocator();
  inf_input_pool = std::make_unique<BatchedBufferPool>(
      batch_size, inf_input_template, *inf_input_allocator);
  inf_output_pool = std::make_unique<BatchedBufferPool>(
      batch_size, inf_output_template, *inf_output_allocator);

  threads.emplace_back(&AxInferenceNet::inference_thread, this, batch_size);
  push_thread = std::jthread(&AxInferenceNet::finalize_thread, this);
  auto inf_allocator = properties.dmabuf_inputs ? create_dma_buf_allocator() :
                                                  create_heap_allocator();
  pre_ops.set_allocator(batch_size, std::move(inf_allocator));
}

void
AxInferenceNet::end_of_input()
{
  auto frame = new_frame();
  frame->video = AxVideoInterface{};
  frame->meta_map = nullptr;
  frame->stream_id = 0;
  frame->frame_id = -2;
  frame->buffer_handle.reset();
  frame->op_input.reset();
  frame->end_of_input = true;
  frame->timestamp = std::chrono::high_resolution_clock::now();
  frame->latency_start = std::chrono::high_resolution_clock::now();
  pre_ops.input_queue().push(std::move(frame));
}

void
default_latency_callback(const std::string &, uint64_t, uint64_t)
{
}

} // namespace Ax

std::unique_ptr<Ax::InferenceNet>
Ax::create_inference_net(const InferenceNetProperties &properties, Ax::Logger &logger,
    InferenceDoneCallback done_callback, LatencyCallback latency_callback)
{
  auto lcb = latency_callback ? latency_callback : default_latency_callback;
  return std::make_unique<AxInferenceNet>(properties, logger, done_callback, lcb);
}

static bool
as_bool(const std::string &s)
{
  return s == "1" || s == "true" || s == "True";
}

Ax::InferenceNetProperties
Ax::read_inferencenet_properties(std::istream &s, Ax::Logger &logger)
{
  InferenceNetProperties props;
  std::string line;
  while (std::getline(s, line)) {
    if (line.empty()) {
      continue;
    }
    auto pos = line.find('=');
    if (pos == std::string::npos) {
      logger(AX_WARN) << "Invalid line in InferenceNetProperties: " << line << std::endl;
      continue;
    }

    const auto key = std::string{ line.substr(0, pos) };
    const auto value = std::string{ line.substr(pos + 1) };
    try {
      if (key == "model") {
        props.model = value;
      } else if (key == "double_buffer") {
        props.double_buffer = as_bool(value);
      } else if (key == "dmabuf_inputs") {
        props.dmabuf_inputs = as_bool(value);
      } else if (key == "dmabuf_outputs") {
        props.dmabuf_outputs = as_bool(value);
      } else if (key == "num_children") {
        props.num_children = std::stoi(value);
      } else if (key == "inference_skip_rate") {
        const auto rate = Ax::parse_skip_rate(value);
        props.skip_stride = rate.stride;
        props.skip_count = rate.count;
      } else if (key == "options") {
        props.options = value;
      } else if (key == "meta") {
        props.meta = value;
      } else if (key == "devices") {
        props.devices = value;
      } else if (key.starts_with("preprocess") || key.starts_with("postprocess")) {
        auto *ops = key.starts_with("preprocess") ? props.preproc : props.postproc;
        const auto under = key.find("_");
        const auto prefixlen = key.starts_with("preprocess") ? 10 : 11;
        const auto num = std::stoi(key.substr(prefixlen, under - prefixlen));
        if (under == std::string::npos || num >= MAX_OPERATORS) {
          logger(AX_WARN) << "Invalid operator keyname in InferenceNetProperties: " << key
                          << std::endl;
          continue;
        }
        const auto subkey = key.substr(under + 1);
        if (subkey == "lib") {
          ops[num].lib = value;
        } else if (subkey == "options") {
          ops[num].options = value;
        } else if (subkey == "mode") {
          ops[num].mode = value;
        } else if (subkey == "batch") {
          ops[num].batch = value;
        } else {
          logger(AX_WARN) << "Invalid operator subkey in InferenceNetProperties: " << key
                          << std::endl;
        }
      } else {
        logger(AX_WARN) << "Invalid key in InferenceNetProperties: " << key << std::endl;
      }
    }

    catch (const std::exception &e) {
      logger(AX_ERROR) << "Failed to convert value for " << key << " : "
                       << e.what() << std::endl;
    }
  }
  return props;
}

Ax::InferenceNetProperties
Ax::read_inferencenet_properties(const std::string &path, Ax::Logger &logger)
{
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return read_inferencenet_properties(f, logger);
}
