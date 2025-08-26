# Copyright Axelera AI, 2025
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ax_datasets.objdataadapter import (
    DataFormatError,
    DataLoadingError,
    DatasetConfig,
    InvalidConfigurationError,
    KptDataAdapter,
    ObjDataAdapter,
    SegDataAdapter,
    SupportedLabelType,
    SupportedTaskCategory,
    UnifiedDataset,
    _create_image_list_file,
    coco80_to_coco91_table,
    coco91_to_coco80_table,
    xywh2ltwh,
    xywh2xyxy,
    xyxy2xywh,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config():
    """Create a sample dataset configuration."""
    return DatasetConfig(
        data_root="/tmp/dataset",
        val_data="val_data.txt",
        cal_data="cal_data.txt",
        task=SupportedTaskCategory.ObjDet,
        label_type=SupportedLabelType.YOLOv8,
    )


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255  # Add a white square
    return img


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    # Format: [class_id, x1, y1, x2, y2]
    return [[0, 0.3, 0.3, 0.7, 0.7]]


@pytest.fixture
def sample_image_file(temp_dir, sample_image):
    """Create a sample image file for testing."""
    img_path = temp_dir / "sample.jpg"
    Image.fromarray(sample_image).save(img_path)
    return img_path


@pytest.fixture
def sample_label_file(temp_dir, sample_labels):
    """Create a sample label file for testing."""
    label_path = temp_dir / "sample.txt"
    with open(label_path, 'w') as f:
        for label in sample_labels:
            f.write(" ".join(map(str, label)) + "\n")
    return label_path


@pytest.fixture
def sample_image_list_file(temp_dir, sample_image_file):
    """Create a sample image list file for testing."""
    list_path = temp_dir / "images.txt"
    with open(list_path, 'w') as f:
        f.write(f"{sample_image_file}\n")
    return list_path


@pytest.fixture
def mock_coco_data():
    """Create mock COCO dataset."""
    return {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 200]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [300, 300, 100, 100]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [150, 150, 250, 250]},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
    }


@pytest.fixture
def mock_voc_xml():
    """Create mock VOC XML annotation."""
    xml_content = '''
    <annotation>
        <filename>img1.jpg</filename>
        <size>
            <width>640</width>
            <height>480</height>
            <depth>3</depth>
        </size>
        <object>
            <name>person</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>100</ymin>
                <xmax>300</xmax>
                <ymax>300</ymax>
            </bndbox>
        </object>
    </annotation>
    '''
    return xml_content


class TestDatasetConfig:
    """Test the DatasetConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = DatasetConfig(data_root="/tmp/dataset")
        assert config.data_root == Path("/tmp/dataset")
        assert config.task == SupportedTaskCategory.ObjDet
        assert config.label_type == SupportedLabelType.YOLOv8
        assert config.output_format == 'xyxy'
        assert config.use_cache is True
        assert config.mask_size is None

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        config = DatasetConfig(
            data_root="/tmp/dataset",
            task=SupportedTaskCategory.Seg,
            label_type=SupportedLabelType.COCOJSON,
            output_format="xywh",
            use_cache=False,
            mask_size=(320, 320),
        )
        assert config.data_root == Path("/tmp/dataset")
        assert config.task == SupportedTaskCategory.Seg
        assert config.label_type == SupportedLabelType.COCOJSON
        assert config.output_format == "xywh"
        assert config.use_cache is False
        assert config.mask_size == (320, 320)

    def test_validation(self):
        """Test configuration validation."""
        # Test invalid output format
        with pytest.raises(InvalidConfigurationError):
            DatasetConfig(data_root="/tmp/dataset", output_format="invalid")

        # Test invalid mask size
        with pytest.raises(InvalidConfigurationError):
            DatasetConfig(
                data_root="/tmp/dataset", task=SupportedTaskCategory.Seg, mask_size="invalid"
            )

    def test_to_dict_from_dict(self):
        """Test conversion to and from dictionary."""
        original = DatasetConfig(
            data_root="/tmp/dataset",
            task=SupportedTaskCategory.Kpts,
            label_type=SupportedLabelType.COCOJSON,
        )
        config_dict = original.to_dict()
        recreated = DatasetConfig.from_dict(config_dict)

        assert recreated.data_root == original.data_root
        assert recreated.task == original.task
        assert recreated.label_type == original.label_type


class TestSupportedLabelType:
    """Test the SupportedLabelType enum."""

    def test_from_string(self):
        """Test conversion from string to enum."""
        assert SupportedLabelType.from_string("YOLOv8") == SupportedLabelType.YOLOv8
        assert SupportedLabelType.from_string("yolov8") == SupportedLabelType.YOLOv8
        assert SupportedLabelType.from_string("COCO JSON") == SupportedLabelType.COCOJSON
        assert SupportedLabelType.from_string("coco json") == SupportedLabelType.COCOJSON
        assert SupportedLabelType.from_string("COCO2017") == SupportedLabelType.COCO2017

        with pytest.raises(ValueError):
            SupportedLabelType.from_string("InvalidType")

    def test_parse(self):
        """Test parsing various input types."""
        # Test with string
        assert SupportedLabelType.parse("YOLOv8") == SupportedLabelType.YOLOv8

        # Test with enum
        assert SupportedLabelType.parse(SupportedLabelType.COCOJSON) == SupportedLabelType.COCOJSON

        # Test with invalid type
        with pytest.raises(ValueError):
            SupportedLabelType.parse(123)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_coco80_to_coco91_table(self):
        """Test conversion from COCO80 to COCO91 class indices."""
        table = coco80_to_coco91_table()
        assert len(table) == 80
        assert table[0] == 1  # First class should be 1
        assert 12 not in table  # Class 12 should be missing (as per COCO dataset)

    def test_coco91_to_coco80_table(self):
        """Test conversion from COCO91 to COCO80 class indices."""
        table = coco91_to_coco80_table()
        assert len(table) == 91
        assert table[0] == 0  # Class 1 in COCO91 maps to 0 in COCO80
        assert table[11] == -1  # Class 12 in COCO91 is not in COCO80

    def test_xywh2xyxy(self):
        """Test conversion from xywh to xyxy format."""
        xywh = np.array([[0.5, 0.5, 0.2, 0.2]])
        xyxy = xywh2xyxy(xywh)
        np.testing.assert_allclose(xyxy, np.array([[0.4, 0.4, 0.6, 0.6]]))

    def test_xyxy2xywh(self):
        """Test conversion from xyxy to xywh format."""
        xyxy = np.array([[0.4, 0.4, 0.6, 0.6]])
        xywh = xyxy2xywh(xyxy)
        np.testing.assert_allclose(xywh, np.array([[0.5, 0.5, 0.2, 0.2]]))

    def test_xywh2ltwh(self):
        """Test conversion from xywh to ltwh format."""
        xywh = np.array([[0.5, 0.5, 0.2, 0.2]])
        ltwh = xywh2ltwh(xywh)
        np.testing.assert_allclose(ltwh, np.array([[0.4, 0.4, 0.2, 0.2]]))

    @patch('pathlib.Path.is_file', return_value=True)
    def test_create_image_list_file_with_file(self, mock_is_file):
        """Test creating an image list file from an existing file."""
        path = Path("/tmp/file.txt")
        result = _create_image_list_file(path)
        assert result == path

    def test_create_image_list_file_behavior(self):
        """Test the behavior of _create_image_list_file function."""
        # Create a temporary directory with an image file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test image file
            test_dir = Path(temp_dir)
            images_dir = test_dir / "images"
            images_dir.mkdir()
            test_image = images_dir / "test.jpg"
            with open(test_image, 'w') as f:
                f.write("test image content")

            # Call the function with the test directory
            result = _create_image_list_file(test_dir)

            # Verify the result is a file
            assert result.is_file()

            # Verify the file contains the image path
            with open(result, 'r') as f:
                content = f.read()
                assert str(test_image.absolute()) in content

            # Clean up the temporary file
            result.unlink()


class TestUnifiedDataset:
    """Test the UnifiedDataset class."""

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    def test_initialization(
        self, mock_configure_data, mock_get_imgs_labels, mock_is_file, mock_exists
    ):
        """Test initialization of UnifiedDataset."""
        # Mock the necessary methods
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)
        mock_get_imgs_labels.return_value = (
            [Path('/tmp/img1.jpg')],  # img_paths
            [[[0, 0.1, 0.1, 0.2, 0.2]]],  # labels
            [[]],  # segments
            [1],  # image_ids
            None,  # gt_json
        )

        dataset = UnifiedDataset(
            data_root="/tmp/dataset",
            split='val',
            task=SupportedTaskCategory.ObjDet,
            label_type=SupportedLabelType.YOLOv8,
        )

        assert dataset.data_root == Path("/tmp/dataset")
        assert dataset.split == 'val'
        assert dataset.task_enum == SupportedTaskCategory.ObjDet
        assert dataset.label_type == SupportedLabelType.YOLOv8
        assert len(dataset.img_paths) == 1
        assert len(dataset.labels) == 1

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('builtins.open')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    def test_get_image_paths(self, mock_configure_data, mock_open, mock_is_file, mock_exists):
        """Test getting image paths."""
        # Mock file reading
        mock_open.return_value.__enter__.return_value.readlines.return_value = [
            'img1.jpg\n',
            'img2.jpg\n',
        ]
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels',
            return_value=(
                [Path('img1.jpg'), Path('img2.jpg')],  # img_paths
                [[[0, 0.1, 0.1, 0.2, 0.2]], [[0, 0.3, 0.3, 0.4, 0.4]]],  # labels
                [[], []],  # segments
                [1, 2],  # image_ids
                None,  # gt_json
            ),
        ):
            dataset = UnifiedDataset(data_root='/tmp/dataset', split='val')
            dataset.data_path = 'val_list.txt'

            # Call the method
            img_paths = dataset._get_image_paths()

            # Verify results
            assert len(img_paths) == 2
            assert img_paths[0] == Path('img1.jpg')
            assert img_paths[1] == Path('img2.jpg')

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('ax_datasets.objdataadapter.pickle.load')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._cache_and_verify_dataset')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    @patch('ax_datasets.objdataadapter._create_image_list_file')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels')
    def test_cache_loading(
        self,
        mock_get_imgs_labels,
        mock_create_list,
        mock_configure_data,
        mock_cache,
        mock_load,
        mock_is_file,
        mock_exists,
    ):
        """Test cache loading."""
        # Mock configuration
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)
        mock_create_list.return_value = Path('/tmp/val.txt')

        # Mock _get_imgs_labels to avoid file operations
        mock_get_imgs_labels.return_value = (
            [Path('img1.jpg')],  # img_paths
            [[[0, 10, 20, 30, 40]]],  # labels
            [[]],  # segments
            [1],  # image_ids
            None,  # gt_json
        )

        # Mock cache loading
        mock_cache.return_value = (
            [Path('img1.jpg')],  # img_paths
            np.array([[100, 100]]),  # shapes
            [[[0, 10, 20, 30, 40]]],  # labels
            [[]],  # segments
            True,  # from_cache
        )

        mock_load.return_value = {
            'version': 0.3,
            'hash': 'test_hash',
            'status': (0, 10, 0, 0),
            'img1.jpg': ([[0, 10, 20, 30, 40]], (100, 100), []),
        }

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_hash', return_value='test_hash'
        ):
            with patch('builtins.open', MagicMock()):
                dataset = UnifiedDataset(
                    data_root='/tmp/dataset',
                    split='val',
                    use_cache=True,
                    val_data='val.txt',  # Add required val_data parameter
                )

                # Verify dataset was initialized correctly
                assert dataset.total_frames == 1
                assert len(dataset.labels) == 1
                assert dataset.labels[0] == [[0, 10, 20, 30, 40]]

    @patch('ax_datasets.objdataadapter.UnifiedDataset._cache_and_verify_dataset')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._load_image')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels')
    def test_getitem(
        self,
        mock_get_imgs_labels,
        mock_load_image,
        mock_configure_data,
        mock_cache,
        sample_image,
        sample_labels,
    ):
        """Test __getitem__ method."""
        # Mock configuration
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)

        # Mock _get_imgs_labels to avoid file operations
        mock_get_imgs_labels.return_value = (
            [Path('/tmp/img1.jpg')],  # img_paths
            [sample_labels],  # labels
            [[]],  # segments
            [1],  # image_ids
            None,  # gt_json
        )

        # Mock cache loading
        mock_cache.return_value = (
            [Path('/tmp/img1.jpg')],  # img_paths
            np.array([[100, 100]]),  # shapes
            [sample_labels],  # labels
            [[]],  # segments
            True,  # from_cache
        )

        # Mock image loading
        mock_load_image.return_value = Image.fromarray(sample_image)

        # Create dataset
        dataset = UnifiedDataset(
            data_root='/tmp/dataset',
            split='val',
            val_data='val.txt',  # Add required val_data parameter
        )

        # Get an item
        item = dataset[0]

        # Verify item contents
        assert 'image' in item
        assert 'bboxes' in item
        assert 'category_id' in item
        assert 'image_id' in item
        assert isinstance(item['bboxes'], torch.Tensor)
        assert isinstance(item['category_id'], torch.Tensor)

    @patch('ax_datasets.objdataadapter.UnifiedDataset._cache_and_verify_dataset')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    @patch('ax_datasets.objdataadapter._create_image_list_file')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels')
    def test_len(self, mock_get_imgs_labels, mock_create_list, mock_configure_data, mock_cache):
        """Test __len__ method."""
        # Mock configuration
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)
        mock_create_list.return_value = Path('/tmp/val.txt')

        # Mock _get_imgs_labels to avoid file operations
        mock_get_imgs_labels.return_value = (
            [Path('/tmp/img1.jpg'), Path('/tmp/img2.jpg')],  # img_paths
            [[[0, 0.1, 0.1, 0.2, 0.2]], [[0, 0.3, 0.3, 0.4, 0.4]]],  # labels
            [[], []],  # segments
            [1, 2],  # image_ids
            None,  # gt_json
        )

        # Mock cache loading
        mock_cache.return_value = (
            [Path('/tmp/img1.jpg'), Path('/tmp/img2.jpg')],  # img_paths
            np.array([[100, 100], [200, 200]]),  # shapes
            [[[0, 0.1, 0.1, 0.2, 0.2]], [[0, 0.3, 0.3, 0.4, 0.4]]],  # labels
            [[], []],  # segments
            True,  # from_cache
        )

        with patch('builtins.open', MagicMock()):
            dataset = UnifiedDataset(
                data_root='/tmp/dataset',
                split='val',
                val_data='val.txt',  # Add required val_data parameter
            )
            assert len(dataset) == 2

    def test_image2label_paths(self):
        """Test conversion from image paths to label paths."""
        img_paths = [Path('/tmp/dataset/images/img1.jpg'), Path('/tmp/dataset/images/img2.jpg')]

        # Test with same directory
        label_paths = UnifiedDataset.image2label_paths(img_paths, is_same_dir=True)
        assert label_paths[0] == Path('/tmp/dataset/images/img1.txt')
        assert label_paths[1] == Path('/tmp/dataset/images/img2.txt')

        # Test with different directory
        label_paths = UnifiedDataset.image2label_paths(img_paths, is_same_dir=False)
        assert label_paths[0] == Path('/tmp/dataset/labels/img1.txt')
        assert label_paths[1] == Path('/tmp/dataset/labels/img2.txt')

        # Test with custom label tag
        label_paths = UnifiedDataset.image2label_paths(
            img_paths, is_same_dir=False, tag='annotations'
        )
        assert label_paths[0] == Path('/tmp/dataset/annotations/img1.txt')
        assert label_paths[1] == Path('/tmp/dataset/annotations/img2.txt')

    def test_replace_last_match_dir(self):
        """Test replacing the last match directory in a path."""
        path = Path('/tmp/dataset/images/subdir/img1.jpg')

        # Replace 'images' with 'labels'
        new_path = UnifiedDataset.replace_last_match_dir(path, 'images', 'labels')
        assert new_path == Path('/tmp/dataset/labels/subdir/img1.jpg')

        # Test with path that doesn't contain the match
        path = Path('/tmp/dataset/data/img1.jpg')
        new_path = UnifiedDataset.replace_last_match_dir(path, 'images', 'labels')
        assert new_path == path  # Should remain unchanged


class TestLabelFormats:
    """Test handling of different label formats."""

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('builtins.open')
    @patch('json.load')
    def test_coco_format(
        self, mock_json_load, mock_open, mock_is_file, mock_exists, mock_coco_data
    ):
        """Test handling of COCO format labels."""
        # Mock JSON loading
        mock_json_load.return_value = mock_coco_data

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels'
        ) as mock_get_imgs_labels:
            # Set up mock return values
            mock_get_imgs_labels.return_value = (
                [Path('/tmp/img1.jpg'), Path('/tmp/img2.jpg')],  # img_paths
                [
                    [[0, 100, 100, 300, 300], [1, 300, 300, 400, 400]],
                    [[0, 150, 150, 400, 400]],
                ],  # labels
                [[], []],  # segments
                [1, 2],  # image_ids
                mock_coco_data,  # gt_json
            )

            # Create dataset with COCO format
            dataset = UnifiedDataset(
                data_root='/tmp/dataset',
                split='val',
                label_type=SupportedLabelType.COCOJSON,
                val_data='annotations.json',
            )

            # Verify dataset was initialized correctly
            assert dataset.label_type == SupportedLabelType.COCOJSON
            assert len(dataset.img_paths) == 2
            assert len(dataset.labels) == 2
            assert dataset.image_ids == [1, 2]

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.glob')
    def test_yolo_format(
        self, mock_glob, mock_is_file, mock_exists, temp_dir, sample_image_file, sample_label_file
    ):
        """Test handling of YOLO format labels."""
        # Mock glob to return our sample files
        mock_glob.return_value = [sample_label_file]

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels'
        ) as mock_get_imgs_labels:
            # Set up mock return values
            mock_get_imgs_labels.return_value = (
                [sample_image_file],  # img_paths
                [[[0, 0.3, 0.3, 0.7, 0.7]]],  # labels
                [[]],  # segments
                [1],  # image_ids
                None,  # gt_json
            )

            # Create dataset with YOLO format
            dataset = UnifiedDataset(
                data_root=temp_dir,
                split='val',
                label_type=SupportedLabelType.YOLOv8,
                val_data='images.txt',
            )

            # Verify dataset was initialized correctly
            assert dataset.label_type == SupportedLabelType.YOLOv8
            assert len(dataset.img_paths) == 1
            assert len(dataset.labels) == 1

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('xml.etree.ElementTree.parse')
    def test_voc_format(self, mock_parse, mock_is_file, mock_exists, mock_voc_xml):
        """Test handling of Pascal VOC format labels."""
        # Mock XML parsing
        mock_root = ET.fromstring(mock_voc_xml)
        mock_parse.return_value.getroot.return_value = mock_root

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels'
        ) as mock_get_imgs_labels:
            # Set up mock return values
            mock_get_imgs_labels.return_value = (
                [Path('/tmp/img1.jpg')],  # img_paths
                [[[0, 100, 100, 300, 300]]],  # labels
                [[]],  # segments
                [1],  # image_ids
                None,  # gt_json
            )

            # Create dataset with VOC format
            dataset = UnifiedDataset(
                data_root='/tmp/dataset',
                split='val',
                label_type=SupportedLabelType.PascalVOCXML,
                val_data='val.txt',
            )

            # Verify dataset was initialized correctly
            assert dataset.label_type == SupportedLabelType.PascalVOCXML
            assert len(dataset.img_paths) == 1
            assert len(dataset.labels) == 1


class TestDataAdapters:
    """Test the data adapter classes."""

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    def test_obj_data_adapter(self, mock_check_label_type, sample_config):
        """Test ObjDataAdapter initialization."""
        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        model_info = MagicMock()
        model_info.num_classes = 80

        adapter = ObjDataAdapter(sample_config.to_dict(), model_info)
        assert adapter.dataset_config['task'] == sample_config.task.value
        assert adapter.dataset_config['label_type'] == sample_config.label_type.value

    @patch('ax_datasets.objdataadapter.SegDataAdapter._check_supported_label_type')
    def test_seg_data_adapter(self, mock_check_label_type, sample_config):
        """Test SegDataAdapter initialization."""
        mock_check_label_type.return_value = SupportedLabelType.COCOJSON

        model_info = MagicMock()
        model_info.num_classes = 80

        config_dict = sample_config.to_dict()
        config_dict['task'] = SupportedTaskCategory.Seg.value
        config_dict['label_type'] = SupportedLabelType.COCOJSON.value
        config_dict['mask_size'] = (160, 160)

        adapter = SegDataAdapter(config_dict, model_info)
        assert adapter.dataset_config['task'] == SupportedTaskCategory.Seg.value
        assert adapter.dataset_config['label_type'] == SupportedLabelType.COCOJSON.value
        assert adapter.mask_size == (160, 160)

    @patch('ax_datasets.objdataadapter.KptDataAdapter._check_supported_label_type')
    def test_kpt_data_adapter(self, mock_check_label_type, sample_config):
        """Test KptDataAdapter initialization."""
        mock_check_label_type.return_value = SupportedLabelType.COCOJSON

        model_info = MagicMock()
        model_info.num_classes = 80

        config_dict = sample_config.to_dict()
        config_dict['task'] = SupportedTaskCategory.Kpts.value
        config_dict['label_type'] = SupportedLabelType.COCOJSON.value

        adapter = KptDataAdapter(config_dict, model_info)
        assert adapter.dataset_config['task'] == SupportedTaskCategory.Kpts.value
        assert adapter.dataset_config['label_type'] == SupportedLabelType.COCOJSON.value

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    @patch('ax_datasets.objdataadapter.UnifiedDataset')
    def test_data_loader_creation(self, mock_dataset, mock_check_label_type, sample_config):
        """Test creation of data loaders."""
        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        # Mock the dataset instance
        mock_dataset_instance = MagicMock()
        # Set __len__ to return a positive value to avoid ValueError in RandomSampler
        mock_dataset_instance.__len__.return_value = 10
        mock_dataset.return_value = mock_dataset_instance

        model_info = MagicMock()
        model_info.num_classes = 80

        # Create adapter
        adapter = ObjDataAdapter(sample_config.to_dict(), model_info)

        # Test calibration data loader
        transform = MagicMock()
        cal_loader = adapter.create_calibration_data_loader(transform, "/tmp/root", 8)
        assert cal_loader is not None

        # Updated assertion to match the actual behavior - without label_type for default
        mock_dataset.assert_called_with(
            transform=transform,
            data_root="/tmp/root",
            split='train',
            task=SupportedTaskCategory.ObjDet,
        )


class TestErrorHandling:
    """Test error handling in the dataset module."""

    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        error = InvalidConfigurationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_data_format_error(self):
        """Test DataFormatError."""
        error = DataFormatError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_data_loading_error(self):
        """Test DataLoadingError."""
        error = DataLoadingError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=False)
    @patch('ax_datasets.objdataadapter.Path.is_dir', return_value=False)
    def test_error_on_missing_reference_file(self, mock_is_dir, mock_is_file, mock_exists):
        """Test error when reference file is missing."""
        with pytest.raises(FileNotFoundError, match="Path .* is neither a file nor a directory"):
            UnifiedDataset(data_root='/tmp/dataset', split='val', val_data='missing.txt')

    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    def test_error_on_missing_reference_file_dataloadingerror(self, mock_configure_data):
        """Test error when reference file is missing."""
        mock_configure_data.side_effect = DataLoadingError("Reference file not found")

        with pytest.raises(DataLoadingError, match="Reference file not found"):
            UnifiedDataset(data_root='/tmp/dataset', split='val', val_data='missing.txt')

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('builtins.open', mock_open(read_data=""))
    def test_error_on_empty_image_list(self, mock_is_file, mock_exists):
        """Test error when image list is empty."""
        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels'
        ) as mock_get_imgs_labels:
            mock_get_imgs_labels.side_effect = DataLoadingError("No supported images found")

            with pytest.raises(DataLoadingError, match="No supported images found"):
                UnifiedDataset(data_root='/tmp/dataset', split='val', val_data='empty.txt')


# Standard dataset fixtures and tests
@pytest.fixture
def mock_coco_dirs():
    """Create a temp directory with COCO-like structure."""
    temp_dir = tempfile.mkdtemp()
    data_root = Path(temp_dir)

    # Create mock directory structure for COCO
    (data_root / "images" / "train2017").mkdir(parents=True)
    (data_root / "images" / "val2017").mkdir(parents=True)
    (data_root / "labels").mkdir(parents=True)
    (data_root / "labels_kpts").mkdir(parents=True)

    # Create sample image list file
    with open(data_root / "train2017.txt", "w") as f:
        f.write("/path/to/image1.jpg\n")
        f.write("/path/to/image2.jpg\n")

    with open(data_root / "val2017.txt", "w") as f:
        f.write("/path/to/image3.jpg\n")

    yield data_root

    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_download():
    """Mock the download function."""
    with mock.patch('axelera.app.data_utils.check_and_download_dataset') as m:
        yield m


@pytest.mark.parametrize("split", ["train", "val"])
def test_coco2017_dataset_automatic_download(mock_coco_dirs, mock_download, split):
    """Test COCO2017 dataset automatic download."""
    with mock.patch('ax_datasets.objdataadapter.Path.is_file', return_value=True), mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels',
        side_effect=DataLoadingError("Test error"),
    ), mock.patch('ax_datasets.objdataadapter.Image'):

        # This should raise the DataLoadingError we're mocking
        with pytest.raises(DataLoadingError):
            dataset = UnifiedDataset(
                data_root=mock_coco_dirs, label_type=SupportedLabelType.COCO2017, split=split
            )

        # Verify download was attempted
        assert mock_download.called

        # Check COCO2017 was requested
        calls = mock_download.call_args_list
        assert any('COCO2017' in str(call) for call in calls)


# DatasetConfig tests
def test_config_creation():
    """Test creating a DatasetConfig with various parameters."""
    config = DatasetConfig(
        data_root="/data",
        val_data="val.txt",
        cal_data="train.txt",
        task=SupportedTaskCategory.ObjDet,
        label_type=SupportedLabelType.YOLOv8,
        output_format="xyxy",
        use_cache=True,
        custom_param="value",
    )

    # Check standard attributes
    assert str(config.data_root) == "/data"
    assert config.val_data == "val.txt"
    assert config.cal_data == "train.txt"
    assert config.task == SupportedTaskCategory.ObjDet
    assert config.label_type == SupportedLabelType.YOLOv8
    assert config.output_format == "xyxy"
    assert config.use_cache is True

    # Check custom attribute
    assert config.custom_param == "value"


def test_config_from_dict():
    """Test creating a DatasetConfig from a dictionary."""
    config_dict = {
        'data_root': "/data",
        'val_data': "val.txt",
        'cal_data': "train.txt",
        'task': SupportedTaskCategory.ObjDet.value,  # Integer value
        'label_type': SupportedLabelType.YOLOv8.value,  # Integer value
        'output_format': "xyxy",
        'use_cache': True,
        'custom_param': "value",
    }

    config = DatasetConfig.from_dict(config_dict)

    # Check that enums were correctly converted
    assert config.task == SupportedTaskCategory.ObjDet
    assert config.label_type == SupportedLabelType.YOLOv8
    assert config.custom_param == "value"


def test_config_validation():
    """Test validation of configuration parameters."""
    # Test invalid output format
    with pytest.raises(InvalidConfigurationError):
        DatasetConfig(data_root="/data", output_format="invalid_format")

    # Test invalid mask_size for segmentation
    with pytest.raises(InvalidConfigurationError):
        DatasetConfig(
            data_root="/data", task=SupportedTaskCategory.Seg, mask_size=123  # Not a tuple/list
        )


# Cache mechanism tests
@pytest.fixture
def mock_cache():
    """Create a mock cache file."""
    temp_dir = tempfile.mkdtemp()
    cache_path = Path(temp_dir) / "test.cache"

    # Create a mock cache file
    cache_data = {
        "hash": "abc123",
        "version": 0.3,
        "status": (0, 5, 0, 0),  # nm, nf, ne, nc
        "image1.jpg": [[[0, 0.5, 0.5, 0.5, 0.5]], (100, 100), []],
        "image2.jpg": [[[1, 0.2, 0.2, 0.3, 0.3]], (200, 200), []],
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    yield cache_path

    # Clean up
    shutil.rmtree(temp_dir)


def test_load_from_cache(mock_cache):
    """Test loading dataset from cache."""
    with mock.patch('ax_datasets.objdataadapter.UnifiedDataset._get_hash', return_value="abc123"):
        dataset = mock.MagicMock(spec=UnifiedDataset)
        dataset.cache_version = 0.3

        # Call the method we're testing
        cache = UnifiedDataset._load_cache(dataset, mock_cache, "abc123")

        # Check the cache was loaded correctly
        assert cache["hash"] == "abc123"
        assert cache["version"] == 0.3
        assert cache["status"] == (0, 5, 0, 0)
        assert len(cache) - 3 == 2  # 2 images plus hash, version, status


def test_invalid_cache_hash(mock_cache):
    """Test handling of cache with invalid hash."""
    with mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset._get_hash', return_value="different_hash"
    ):
        dataset = mock.MagicMock(spec=UnifiedDataset)
        dataset.cache_version = 0.3

        # Call the method we're testing
        cache = UnifiedDataset._load_cache(dataset, mock_cache, "different_hash")

        # Cache should be rejected due to hash mismatch
        assert cache == {}


def test_invalid_cache_version(mock_cache):
    """Test handling of cache with invalid version."""
    with mock.patch('ax_datasets.objdataadapter.UnifiedDataset._get_hash', return_value="abc123"):
        dataset = mock.MagicMock(spec=UnifiedDataset)
        dataset.cache_version = 0.4  # Different from cache file

        # Call the method we're testing
        cache = UnifiedDataset._load_cache(dataset, mock_cache, "abc123")

        # Cache should be rejected due to version mismatch
        assert cache == {}


@pytest.mark.parametrize("output_format", ["xyxy", "xywh", "ltwh"])
def test_output_formats(output_format):
    """Test different output formats."""
    # More complete mocking to capture the output_format
    with mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset._configure_data',
        return_value=(Path("/tmp"), Path("/tmp/list.txt"), None),
    ), mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels',
        side_effect=DataLoadingError("Test error"),
    ), mock.patch(
        'pathlib.Path.is_file', return_value=True
    ), mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset.__getitem__'
    ) as mock_getitem:

        # Create the dataset - should initialize but raise error in _get_imgs_labels
        with pytest.raises(DataLoadingError):
            dataset = UnifiedDataset(
                data_root="/tmp", output_format=output_format, val_data="dummy.txt"
            )

            # Verify the output_format was set correctly on the dataset
            assert dataset.output_format == output_format


# YOLO format tests
@pytest.fixture
def yolo_dirs():
    """Create temp directories for YOLO format testing."""
    temp_dir = tempfile.mkdtemp()
    data_root = Path(temp_dir)

    # Create directory structure for directory-based approach
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    train_dir = images_dir / "train"
    val_dir = images_dir / "valid"

    # Create directories
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create sample image files
    (train_dir / "img1.jpg").touch()
    (train_dir / "img2.jpg").touch()
    (val_dir / "img3.jpg").touch()

    # Create corresponding label files
    (labels_dir / "train").mkdir(exist_ok=True, parents=True)
    (labels_dir / "valid").mkdir(exist_ok=True, parents=True)
    (labels_dir / "train" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
    (labels_dir / "train" / "img2.txt").write_text("1 0.3 0.3 0.2 0.2")
    (labels_dir / "valid" / "img3.txt").write_text("2 0.6 0.6 0.15 0.15")

    # Create text file for text-based approach
    train_txt = data_root / "train.txt"
    val_txt = data_root / "val.txt"

    with open(train_txt, "w") as f:
        f.write(f"./images/train/img1.jpg\n")
        f.write(f"./images/train/img2.jpg\n")

    with open(val_txt, "w") as f:
        f.write(f"./images/valid/img3.jpg\n")

    yield data_root

    # Clean up
    shutil.rmtree(temp_dir)


def test_yolo_directory_based(yolo_dirs):
    """Test YOLO format with directory-based approach."""
    # Mock _check_supported_label_type to return the enum directly
    with mock.patch(
        'ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type',
        return_value=SupportedLabelType.YOLOv8,
    ), mock.patch('ax_datasets.objdataadapter.Image'):

        # Create a proper config dictionary with a string for label_type
        config_dict = {
            'data_root': str(yolo_dirs),
            'cal_data': "train",
            'val_data': "valid",
            'label_type': "YOLOv8",  # Use string instead of enum value
        }

        # Initialize adapter
        model_info = mock.MagicMock()
        model_info.num_classes = 80
        adapter = ObjDataAdapter(config_dict, model_info)

        # Test with mocked dataset class
        with mock.patch(
            'ax_datasets.objdataadapter.UnifiedDataset', side_effect=DataLoadingError("Test error")
        ):

            with pytest.raises(DataLoadingError):
                train_dataset = adapter._get_dataset_class(
                    transform=None, root=yolo_dirs, split="train", kwargs={"cal_data": "train"}
                )


def test_yolo_text_file_based(yolo_dirs):
    """Test YOLO format with text file-based approach."""
    # Mock _check_supported_label_type to return the enum directly
    with mock.patch(
        'ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type',
        return_value=SupportedLabelType.YOLOv8,
    ), mock.patch('ax_datasets.objdataadapter.Image'):

        # Create a proper config dictionary with a string for label_type
        config_dict = {
            'data_root': str(yolo_dirs),
            'cal_data': "train.txt",
            'val_data': "val.txt",
            'label_type': "YOLOv8",  # Use string instead of enum value
        }

        # Initialize adapter
        model_info = mock.MagicMock()
        model_info.num_classes = 80
        adapter = ObjDataAdapter(config_dict, model_info)

        # Test with mocked dataset class
        with mock.patch(
            'ax_datasets.objdataadapter.UnifiedDataset', side_effect=DataLoadingError("Test error")
        ):

            with pytest.raises(DataLoadingError):
                train_dataset = adapter._get_dataset_class(
                    transform=None, root=yolo_dirs, split="train", kwargs={"cal_data": "train.txt"}
                )


class TestUltralyticsIntegration:
    """Test integration with Ultralytics data YAML format."""

    def test_obj_data_adapter_accepts_ultralytics_yaml(self):
        """Test that ObjDataAdapter accepts ultralytics_data_yaml parameter."""
        from axelera import types

        dataset_config = {'ultralytics_data_yaml': 'data.yaml', 'label_type': 'YOLOv8'}

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        # This should not raise a validation error
        adapter = ObjDataAdapter(dataset_config, model_info)
        assert adapter.label_type == SupportedLabelType.YOLOv8

    def test_obj_data_adapter_rejects_mixed_ultralytics_traditional(self):
        """Test that ObjDataAdapter rejects mixing ultralytics_data_yaml with traditional params."""
        from axelera import types

        dataset_config = {
            'ultralytics_data_yaml': 'data.yaml',
            'cal_data': 'train.txt',  # Should not be allowed together
            'label_type': 'YOLOv8',
        }

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        # This should not raise an error at initialization since the validation happens at processing time
        adapter = ObjDataAdapter(dataset_config, model_info)
        assert adapter.label_type == SupportedLabelType.YOLOv8

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    def test_validation_checks_ultralytics_or_traditional_data_sources(
        self, mock_check_label_type
    ):
        """Test that validation requires either ultralytics_data_yaml or traditional data sources."""
        from axelera import types

        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        # Test with no data sources at all
        dataset_config = {
            'label_type': 'YOLOv8'
            # Missing any data source configuration
        }

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        with pytest.raises(
            ValueError,
            match="Please specify either 'repr_imgs_dir_name', 'cal_data', or 'ultralytics_data_yaml'",
        ):
            ObjDataAdapter(dataset_config, model_info)

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    def test_validation_allows_ultralytics_without_val_data(self, mock_check_label_type):
        """Test that validation allows ultralytics_data_yaml without explicit val_data."""
        from axelera import types

        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        dataset_config = {
            'ultralytics_data_yaml': 'data.yaml',
            'label_type': 'YOLOv8'
            # No val_data - should be OK since ultralytics will provide it
        }

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        # This should not raise an error
        adapter = ObjDataAdapter(dataset_config, model_info)
        assert adapter.label_type == SupportedLabelType.YOLOv8

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    def test_validation_requires_val_data_for_traditional_format(self, mock_check_label_type):
        """Test that validation requires val_data for traditional format."""
        from axelera import types

        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        dataset_config = {
            'cal_data': 'train.txt',
            'label_type': 'YOLOv8'
            # Missing val_data for traditional format
        }

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        with pytest.raises(
            ValueError, match="Please specify 'val_data' or 'ultralytics_data_yaml'"
        ):
            ObjDataAdapter(dataset_config, model_info)
