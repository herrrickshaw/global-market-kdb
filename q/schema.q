/ schema.q - daily OHLCV bars, one HDB partition per trading date.
/ `date` is the partition column, so it is virtual: it is the partition
/ directory name, not a stored column. Rows within a partition are sorted by
/ sym with the parted attribute (applied by .Q.dpft in load.q).

daily:([] sym:`symbol$(); mkt:`symbol$(); open:`float$(); high:`float$(); low:`float$(); close:`float$(); volume:`float$())

/ column types of the interchange CSV written by the Python extractors:
/ date,mkt,sym,open,high,low,close,volume
csvtypes:"DSSFFFFF"
