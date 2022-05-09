# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from pathlib import Path
import time

from fem_quadrilateral import QuadrilateralSolver


def main():
    main_root = Path("QS_mls_order_analysis")
    check_running_folder = main_root / "check_running_folder"
    while check_running_folder.exists():
        print("still running...")
        time.sleep(300)  # wait 5 mins

    pot_max_order = 10
    max_order = -1
    for p_order in range(1, pot_max_order):
        root = main_root / f"p_order_{p_order}"
        print("root:", root)
        try:
            q = QuadrilateralSolver.from_root(root)
            q.matrix_lsq_setup()
            q.matrix_lsq(root)
        except Exception as e:
            # dataset in root is bad.
            # rest does not exist
            max_order = p_order - 1
            break

        from make_plots_QS_snapshots_for_mls_order_analysis import main as main1
        main1(max_order)

        from make_error_plots import main as main_e
        main_e(max_order)

        from make_plots_QS_snapshots_for_mls_order_analysis_2 import main as main2
        main2(max_order)

        from make_pod_plots import main as main_p
        # list of each second order.
        p_order_list = [p for p in range(2, max_order + 1, 2)]
        main_p(p_order_list)


if __name__ == '__main__':
    main()
