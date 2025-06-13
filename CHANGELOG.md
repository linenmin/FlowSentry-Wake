![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Changelog

- [Changelog](#changelog)
  - [[Unreleased]](#unreleased)
    - [Breaking Changes](#breaking-changes)
    - [Added](#added)
    - [Changed](#changed)
    - [Deprecated](#deprecated)
    - [Removed](#removed)
    - [Fixed](#fixed)
    - [Security](#security)
  - [[1.0.0] - 2024-11-01](#100---2024-11-01)
    - [Added](#added)
  - [Usage](#usage)
    - [Pull Request Process](#pull-request-process)
    - [Release Process](#release-process)

All notable changes to this SDK will be documented in this file to assist users in migrating their YAML configurations and pipelines in line with SDK upgrades, ensuring seamless model and pipeline deployment.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Breaking Changes
- To deploy an ONNX model, update the YAML `models` section to use the new path:  
  `$AXELERA_FRAMEWORK/ax_models/base_onnx.py`  
  instead of:  
  `$AXELERA_FRAMEWORK/ax_models/onnx/model.py`.  
  - This change aligns the ONNX model deployment path with the structure used for `base_torch.py`.
- Added variable substitution support for Axelera YAMLs, with the syntax
  being ${{MODEL_INFO_VARIABLE}}
  - Any templates using the old {{MODEL_INFO_VARIABLE}} syntax must prepend
    the `$` to continue working
- `compilation_config` in model YAML is now a flat mapping of fields. The subsections such
   as `backend_config` have been removed.
   - TODO: some field names have changed/been removed, we need to document these and provide a
           migration guide

### Added
- Introduced strictYAML and built a schema for built-in AxOperators to provide clear messages
  for incorrect usage in the YAML pipeline
- display.Window now has a title method that allows a stream to be identified.
  inference.py will set the title if there is more than one stream.

### Changed
- Customer models in the `customers` directory don't require the `c-` prefix anymore before `model_name` when calling `deploy` or `inference` functions


### Deprecated
- Features that will be removed in upcoming releases

### Removed
- Features removed in this release

### Fixed
- Bug fixes

### Security
- Security vulnerability fixes

## [1.0.0] - 2024-11-01
### Added
- Initial release

## Usage
- Start with the [README](/README.md)

### Pull Request Process
1. Every PR should consider updating `CHANGELOG.md`.
2. Add changes under the `[Unreleased]` section.
3. Use appropriate categories.
4. For breaking changes:
   - Add under the "Breaking Changes" category.
   - Include migration instructions if possible.

### Release Process
1. When ready for release:
   - Create a new version section (e.g., `[2.0.0]`).
   - Move items from `[Unreleased]` to the new version.
   - Add the release date.
2. Branch out from `main` as `release/<version>` (e.g., `release/2.0.0`)
3. Remove the `[Unreleased]`, `[Pull Request Process]`, and `[Release Process]` sections.
