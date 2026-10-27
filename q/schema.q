/ schema.q - daily OHLCV bars, one HDB partition per trading date.
/ `date` is the partition column, so it is virtual: it is the partition
/ directory name, not a stored column. Rows within a partition are sorted by
/ sym with the parted attribute (applied by .Q.dpft in load.q).

daily:([] sym:`symbol$(); mkt:`symbol$(); open:`float$(); high:`float$(); low:`float$(); close:`float$(); volume:`float$())

/ column types of the interchange CSV written by the Python extractors:
/ date,mkt,sym,open,high,low,close,volume
csvtypes:"DSSFFFFF"

/ write t (date,mkt,sym,open,high,low,close,volume; any markets) as one HDB
/ partition per date. .Q.dpft enumerates symbol columns against hdb/sym,
/ sorts each partition by sym and sets `p#sym. Re-running overwrites the
/ partitions it writes; the sym file is extended.
writehdb:{[hdb;t]
  {[hdb;t;d] `daily set cols[daily]#delete date from select from t where date=d; .Q.dpft[hdb;d;`sym;`daily]}[hdb;t] each asc distinct t`date;
  `daily set 0#daily;}
