from enum import IntEnum


class ScoringFunc(IntEnum):
    SIGMOID = 0
    SQRTSOFTPLUS = 1
    SOFTMAX = 2
    IDENTITY = 3

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, label: str):
        try:
            return cls[label.upper()]
        except (AttributeError, KeyError):
            raise ValueError(f'{label} is not a valid {cls.__name__}') from None
