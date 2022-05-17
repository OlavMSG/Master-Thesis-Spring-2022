# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from pathlib import Path
import time

from src.fem_quadrilateral import QuadrilateralSolver


def main():
    main_root = Path("QS_mls_order_analysis")
    check_running_folder = main_root / "check_running_folder"
    while check_running_folder.exists():
        print("still running...")
        time.sleep(300)  # wait 5 mins

    for p_order in range(1, 5):
        root = main_root / f"p_order_{p_order}"
        print("root:", root)
        q = QuadrilateralSolver.from_root(root)
        q.matrix_lsq_setup()
        q.matrix_lsq(root)

        from make_plots_QS_snapshots_for_mls_order_analysis import main as main1
        main1()

        from make_error_plots import main as main_e
        main_e()

        from make_plots_QS_snapshots_for_mls_order_analysis_2 import main as main2
        main2()

        from make_pod_plots import main as main_p
        main_p()

        from save_pod_errors import main as main_ps
        main_ps()


if __name__ == '__main__':
    main()
