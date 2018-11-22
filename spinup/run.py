
import tensorflow as tf
from spinup.algos.ddpg import ddpg

def create_runner(args, logger_kwargs):
    if args.algo == 'ddpg':
        return ddpg.Runner(env_name=args.env, epochs=args.epochs, logger_kwargs=logger_kwargs)

def run(args, logger_kwargs):
    tf.logging.set_verbosity(tf.logging.INFO)
    ddpg.load_gin_configs(args.gin_file, args.gin_bindings)
    runner = create_runner(args, logger_kwargs)
    runner.run_experiment()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ddpg')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gin_file', type=str, default=['/home/xff/Code/spinningup/spinup/algos/ddpg/ddpg.gin'])
    parser.add_argument('--gin_bindings', type=str, default=[])
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.algo, seed=args.seed)

    run(args, logger_kwargs)

