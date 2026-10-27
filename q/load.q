/ load.q - interchange CSV -> date-partitioned HDB of daily bars.
/ usage: q q/load.q -csv data/sample/ohlcv.csv -hdb db

o:.Q.opt .z.x
qdir:$[1<count p:"/" vs string .z.f; "/" sv -1_p; "."]
system "l ",qdir,"/schema.q"

if[not all `csv`hdb in key o; -2 "usage: q q/load.q -csv FILE -hdb DIR"; exit 1]
raw:(csvtypes;enlist ",") 0: hsym `$first o`csv
raw:select from raw where close>0
hdb:hsym `$first o`hdb

writehdb[hdb;raw];

-1 "wrote ",string[count raw]," rows, ",string[count distinct raw`sym]," syms, ",string[count distinct raw`date]," date partitions to ",1_string hdb;
exit 0
