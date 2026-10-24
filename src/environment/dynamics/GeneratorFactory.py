import enum

from .IGenerator import IGenerator


class GeneratorType(enum.Enum):
    F = enum.auto()
    DeltaV = enum.auto()
    CW = enum.auto()


class GeneratorFactory(object):
    @staticmethod
    def get_generator(
        generator_type: GeneratorType,
    ) -> IGenerator:
        if generator_type.value == GeneratorType.F.value:
            from .OrbitGeneratorF import OrbitGeneratorF

            return OrbitGeneratorF()
        elif generator_type.value == GeneratorType.DeltaV.value:
            from .OrbitGeneratorDeltaV import OrbitGeneratorDeltaV

            return OrbitGeneratorDeltaV()
        elif generator_type.value == GeneratorType.CW.value:
            from .OrbitGeneratorCW import OrbitGeneratorCW

            return OrbitGeneratorCW()
        else:
            raise NotImplementedError()
