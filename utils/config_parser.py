import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--model_id', type=str, required=True,
                        default='test', help='model id')
    parser.add_argument('--model_comment', type=str, required=True,
                        default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True,
                        help='model name')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True,
                        default='GBPCAD', help='dataset type')
    parser.add_argument('--root_path', type=str,
                        default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str,
                        default='gbpcad_one_hour_202311210827.csv', help='data file')
    parser.add_argument('--target', type=str, default='close',
                        help='target feature')
    parser.add_argument('--loader', type=str,
                        default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                        'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                        'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--num_lags', type=int, default=5, help='number of lags columns')
    parser.add_argument('--num_entries', type=int, help='number of entries to take from dataset')
    parser.add_argument('--pred_len', type=int)
    parser.add_argument('--seq_step', type=int)

    # optimization
    parser.add_argument('--num_workers', type=int,
                        default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int,
                        default=8, help='batch size of model evaluation')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')
    parser.add_argument('--pct_start', type=float,
                        default=0.2, help='pct_start')

    return parser.parse_args()
