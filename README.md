# ML_electronic_coupling

build with `python setup.py build_ext --inplace`

examples:
---------
AOM:
python ecp.py -c -9129.69 --mo 98 -p examples/rubrene/AOM/AOM_COEFF.dat --mode aom -o rubrene_aom.dat examples/rubrene/dataset/*.xyz

delta_ml:
python ecp.py -c -9129.69 --mo 98 -p examples/rubrene/AOM/AOM_COEFF.dat --mode delta_ml -o rubrene_aom.dat --cnn_dir examples/rubrene/nnp_correction_to_AOM/model/ examples/rubrene/dataset_xyz/*.xyz

AMD sampling for AOM:
python dimer_sampling_AMD.py -d examples/rubrene/dataset/ -n 100 -oa ~/temp/picked.xxx -or ~/temp/notpicked.xxx
