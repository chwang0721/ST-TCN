import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=3000)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--dataset', type=str, default='sz')

parser.add_argument('--threshold_d', type=int, default=100)
parser.add_argument('--threshold_t', type=int, default=10)

parser.add_argument('--grid_size', type=float, default=0.1)
parser.add_argument('--cpu_num', type=int, default=12)
parser.add_argument('--mode', type=str, default='train')

parser.add_argument('--contact_factor', type=int, default=3)
parser.add_argument('--score_type', type=str, default='s_t')

args = parser.parse_args()
