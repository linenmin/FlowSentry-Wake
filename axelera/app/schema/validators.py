# Copyright Axelera AI, 2025

import re

import strictyaml as sy


class AllowDashValidator(sy.ScalarValidator):
    def validate_scalar(self, chunk):
        return re.sub(r'[-_]', '', chunk.contents)


class CaseInsensitiveEnumValidator(sy.ScalarValidator):
    def __init__(self, values):
        self.values = values
        self.values_lower = [v.lower() for v in values]

    def validate_scalar(self, chunk):
        if chunk.contents.lower() in self.values_lower:
            # Return the original case version
            index = self.values_lower.index(chunk.contents.lower())
            return self.values[index]
        raise sy.exceptions.YAMLValidationError(
            f"when expecting one of {self.values}", f"found '{chunk.contents}'", chunk
        )


class IntEnumValidator(sy.ScalarValidator):
    def __init__(self, values):
        self.values = [int(v) for v in values]

    def validate_scalar(self, chunk):
        try:
            if int(chunk.contents) in self.values:
                return int(chunk.contents)
        finally:
            raise sy.exceptions.YAMLValidationError(
                f"when expecting one of {self.values}", f"found '{chunk.contents}'", chunk
            )
