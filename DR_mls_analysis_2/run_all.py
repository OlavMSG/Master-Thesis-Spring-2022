# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""


def main():
    from save_DR_snapshots_for_mls_order_analysis import main as save_main
    save_main()
    from make_plots_DR_snapshots_for_mls_order_analysis import main as main_a1
    main_a1()
    from make_plots_DR_snapshots_for_mls_order_analysis_2 import main as main_a2
    main_a2()
    from make_error_plots import main as main_error
    main_error()
    from make_pod_plots import main as main_pod
    main_pod()


if __name__ == '__main__':
    main()
