// Copyright Axelera AI, 2025
#pragma once

#include <time.h>

#include <chrono>
#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

class AxMetaStreamId : public AxMetaBase
{
  public:
  int stream_id = 0;
  std::uint64_t timestamp{};

  explicit AxMetaStreamId(int stream_id) : stream_id{ stream_id }
  {
    auto now = std::chrono::high_resolution_clock::now();
    timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch())
                    .count();
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *class_meta = "stream_meta";
    auto results = std::vector<extern_meta>{
      { class_meta, "stream_id", int(sizeof(stream_id)),
          reinterpret_cast<const char *>(&stream_id) },
      { class_meta, "timestamp", int(sizeof(timestamp)),
          reinterpret_cast<const char *>(&timestamp) },
    };
    return results;
  }
};
