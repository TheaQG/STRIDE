
For running the data splits creation scirpt:
`
python -m data_adapters.norcp.launch_splits \
  --root-dir /Users/au728490/Data/NorCP/cropped \
  --scenario-name ECMWF-ERAINT \
  --val-start 2010-01-01T00:00:00 \
  --val-end 2012-12-31T18:00:00 \
  --test-start 2013-01-01T00:00:00 \
  --test-end 2018-12-31T18:00:00
`