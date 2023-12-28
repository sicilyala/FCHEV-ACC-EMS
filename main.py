# main program of training and evaluating
import sys
from common.runner import Runner, set_seed
# from common.runner_PER import Runner
from common.arguments import get_args
from common.env import make_env
from common.utils import Logger
from common.evaluate import Evaluator

if __name__ == '__main__':
    args = get_args()
    env, args = make_env(args)
    if args.evaluate:
        args.scenario_name += '__'+args.load_dir+'_'+args.load_scenario_name+'_%d'%args.load_episode
        sys.stdout = Logger(filepath=args.eva_dir+"/"+args.scenario_name+"/", filename='evaluate_log.log')
        print('number of episode: ', args.evaluate_episode)
    else:
        args.scenario_name += args.file_v
        sys.stdout = Logger(filepath=args.save_dir+"/"+args.scenario_name+"/", filename='train_log.log')
        print('number of episode: ', args.max_episodes)
    print('episode_steps: ', args.episode_steps)
    print('cycle name: ', args.scenario_name)
    print('trip length: %.3fkm' % args.trip_length)
    print('obs_shape: ', args.obs_shape)
    print('action_shape: ', args.action_shape)
    print('reward ratio, ACC: %.1f, EMS: %.1f' % (args.w_ACC, args.w_EMS))

    if args.evaluate:
        print("-----Start evaluating!-----")
        evaluator = Evaluator(args, env)
        evaluator.evaluate()
        print("-----Evaluating is finished!-----")
        print('-----Data saved in: <%s>-----'%(args.eva_dir+"/"+args.scenario_name))
    else:
        print("-----Start training-----")
        set_seed()
        runner = Runner(args, env)
        runner.run()
        print("-----Training is finished!-----")
        print('-----Data saved in: <%s>-----'%(args.save_dir+"/"+args.scenario_name))
