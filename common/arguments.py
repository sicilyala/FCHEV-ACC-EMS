import argparse

"""
Here are the parameters for training
dataset use:
    debug: Standard_IM240, 241s
    train: WVU_mix, 4713s; mix2, 4619s; mix3, 2431s; mix_city 3812s;
    validation: Standard_ChinaCity (CTUDC), 1314s; CLTC_P, 1800s; mix_valid; Standard_HWFET
    # Standard_ChinaCity  Standard_IM240, CLTC_P, Standard_UDDS,
      Standard_WVUCITY, CYC_NewYorkBus, CYC_NYCC, Standard_JN1015
"""


def get_args():
    parser = argparse.ArgumentParser("DDPG-torch")
    # environment
    parser.add_argument('--max_episodes', type=int, default=500)
    # replay buffer
    parser.add_argument('--buffer-size', type=int, default=3e4)  # 3e4 for 500 episode
    parser.add_argument('--batch-size', type=int, default=64)
    # core training parameters
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--lr_base", type=float, default=5e-5, help="learning rate of actor")
    parser.add_argument("--noise_rate", type=float, default=0.25,
                        help="initial noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--noise_discount_rate", type=float, default=0.999)
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="parameter for updating target network")
    # save model under training
    parser.add_argument("--scenario_name", type=str, default="Standard_WVUCITY", help="name of driving cycle data")
    parser.add_argument("--save_dir", type=str, default="./result1",
                        help="directory in which saves training data and model")
    parser.add_argument("--w_ACC", type=float, default=1, help="ACC-reward 权重，固定")
    parser.add_argument("--w_EMS", type=float, default=10, help="EMS-reward 权重")
    parser.add_argument("--file_v", type=str, default='_v1', help="每次训练都须重新指定")
    # load learned model to train new model or evaluate
    parser.add_argument("--load_or_not", type=bool, default=False)
    parser.add_argument("--load_episode", type=int, default=899)
    parser.add_argument("--load_scenario_name", type=str, default="")
    parser.add_argument("--load_dir", type=str, default="./result2")
    # evaluate
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--evaluate_episode", type=int, default=1)
    parser.add_argument("--eva_dir", type=str, default="./validation",
                        help="directory in which saves evaluation result")
    # all above
    args = parser.parse_args()
    return args
