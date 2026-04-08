from abc import ABC, abstractmethod
import logging
import multiprocessing as mp
from pathlib import Path
import time
from typing import Dict, List, Union, Tuple, TypeVar

from pydantic import BaseModel

from config import generate_argparse, load_configs, TrainConfig
from train_t5 import train


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PARAMETER_TYPES = Union[str, int, float]
PARAMETER_TYPE = TypeVar('PARAMETER_TYPE', bound=PARAMETER_TYPES)


class GridConfig(BaseModel):
    variables: Dict[str, Union[PARAMETER_TYPES, List[PARAMETER_TYPES]]]


class Value(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def next(self) -> Tuple[PARAMETER_TYPE, bool]:
        pass

    @abstractmethod
    def get_current(self) -> PARAMETER_TYPE:
        pass


class StaticValue(Value):
    def __init__(self, name: str, value: PARAMETER_TYPE):
        super().__init__(name)
        self.value = value

    def next(self) -> Tuple[PARAMETER_TYPE, bool]:
        return self.value, True

    def get_current(self) -> PARAMETER_TYPE:
        return self.value


class LooperValue(Value):
    def __init__(self, name: str, values: List[PARAMETER_TYPE]):
        super().__init__(name)
        self.values = values
        self.current = 0

    def reset(self):
        self.current = 0

    def next(self) -> Tuple[PARAMETER_TYPE, bool]:
        v = self.values[self.current]
        r = False
        self.current += 1
        if self.current >= len(self.values):
            self.current = 0
            r = True
        return v, r

    def get_current(self) -> PARAMETER_TYPE:
        return self.values[self.current]


def parse_search_ctx(search_ctx: GridConfig) -> List[Value]:
    result = []
    for name in search_ctx.variables.keys():
        if isinstance(search_ctx.variables[name], list):
            result.append(LooperValue(name, search_ctx.variables[name]))
        else:
            result.append(StaticValue(name, search_ctx.variables[name]))
    return result


def generate_grid_job(
        base_ctx: TrainConfig,
        parameters: Dict[str, PARAMETER_TYPES],
        *proc_args
) -> mp.Process:
    logger.info(f'Generating grid job with parameters {parameters}')
    current_ctx = base_ctx.model_copy(deep=True)
    current_ctx.model.hyperparameters = parameters
    logger.info(f'Starting grid process')
    proc = mp.Process(target=train, args=(current_ctx, *proc_args))
    proc.start()
    return proc


def perform_ripple(ranges: List[Value]) -> Tuple[Dict[str, PARAMETER_TYPE], bool]:
    result = {}
    resets = []
    needs_next = True
    for v in ranges:
        reset = False
        if needs_next:
            value, reset = v.next()
        else:
            value = v.get_current()
        result[v.name] = value
        resets.append(reset)
        needs_next = reset
    return result, all(resets)


def loop_grid_search(
        base_ctx: TrainConfig,
        search_ctx: GridConfig,
):
    logger.info('parsing search context into ranges')
    ranges = parse_search_ctx(search_ctx)

    procs = []

    logger.info(f'starting grid search')
    while True:
        logger.info('performing ripple update')
        current_config, finished = perform_ripple(ranges)

        logger.info(f'there are currently {len(procs)} jobs running')

        while len(procs) >= base_ctx.grid_search.max_parallel_jobs:
            found_finished = False
            for p in procs[:]:
                if not p.is_alive():
                    logger.info('job has completed')
                    p.join()
                    procs.remove(p)
                    found_finished = True
            if found_finished:
                logger.info(f'there are currently {len(procs)} jobs running')
                continue
            logger.info("max jobs deployed, waiting for resources")
            time.sleep(base_ctx.grid_search.polling_interval)

        logger.info('starting new job')

        proc = generate_grid_job(
            base_ctx, current_config
        )
        procs.append(proc)

        if finished:
            logger.info('all jobs queued, waiting for finish')
            break

    for p in procs:
        p.join()

    logger.info(f'finished generating grid search')


def main():
    ap = generate_argparse('performs grid search on ByT5 trainer')
    ag = ap.add_argument_group('grid search')
    ag.add_argument('-p', '--parameters', type=Path, nargs='*', help='the parameter configs to use')
    ag.add_argument('--default-parameters', type=Path, default=Path('config/parameters/default.json'), help='the default parameters to use')
    args = ap.parse_args()
    base_ctx = load_configs(args.configs, default_config=args.default_config)
    search_ctx = load_configs(args.parameters, default_config=args.default_parameters, schema=GridConfig)
    loop_grid_search(base_ctx, search_ctx)


if __name__ == '__main__':
    main()