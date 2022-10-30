# -*- coding: utf-8 -*-
# @Time   : 2020/8/26 12:08
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : parser_old.py


def parse_args(my_parser, params=None):

    my_parser.add_argument('--embedding_size', type=int, default=96)
    my_parser.add_argument('--ncaps', type=int, default=2)
    my_parser.add_argument('--rs_rout_iteration', type=int, default=1)
    my_parser.add_argument('--rs_nlayer', type=int, default=1)
    my_parser.add_argument('--neighbor_size', type=int, default=16)
    my_parser.add_argument('--kg_nlayer', type=int, default=1)
    my_parser.add_argument('--kg_neighbor_size', type=int, default=8)
    my_parser.add_argument('--kg_weight', type=float, default=0.1)
    my_parser.add_argument('--mim_weight', type=float, default=0.1)
    my_parser.add_argument('--reg_weight', type=float, default=5e-5)
    my_parser.add_argument('--dropout', type=float, default=0.3)
    args, _ = my_parser.parse_known_args()

    return args


def build_info_str(args):
    info_str = 'lr%s_batchsize%d_' \
               'ncaps%d_hidden%d_' \
               'rsroutit%d_rsnlayer%d_rsneib%d_' \
               'kgnlayer%d_kgneib%d_' \
               'kgweight%s_mimweight%s_regweight%s_dropout%s' % \
               (str(args.lr), args.batch_size,
                args.ncaps, args.embedding_size // args.ncaps,
                args.rs_rout_iteration, args.rs_nlayer, args.neighbor_size,
                args.kg_nlayer, args.kg_neighbor_size,
                str(args.kg_weight), str(args.mim_weight), str(args.reg_weight), str(args.dropout))
    return info_str
