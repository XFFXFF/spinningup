
import tensorflow as tf

from spinup.utils.mpi_tools import mpi_fork
from spinup.algos.ddpg import ddpg
from spinup.algos.vpg import vpg


def create_runner(args, logger_kwargs):
    if args.algo == 'ddpg':
        return ddpg.Runner(env_name=args.env, seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs)
    if args.algo == 'vpg':
        mpi_fork(args.cpu)
        return vpg.Runner(env_name=args.env, seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs)

def run(args, logger_kwargs):
    tf.logging.set_verbosity(tf.logging.INFO)
    ddpg.load_gin_configs(args.gin_files, args.gin_bindings)
    runner = create_runner(args, logger_kwargs)
    if args.test:
        runner.run_test_and_render()
    else:
        runner.run_experiment()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ddpg')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gin_files', nargs='+', default=["spinup/algos/ddpg/ddpg.gin"])
    parser.add_argument('--gin_bindings', nargs='+', default=[])
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.algo, env_name=args.env, seed=args.seed)

    run(args, logger_kwargs)

