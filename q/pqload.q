/ pqload.q - Parquet warehouse -> date-partitioned HDB, read natively with the
/ KDB-X pq module (no Python, no CSV). Reads <root>/<MKT>/year=YYYY.parquet
/ (columns Date,Symbol,Open,High,Low,Close,Volume), keeps the `-names` most
/ liquid symbols per market by median dollar volume over the window, and writes
/ the same HDB as load.q.
/ usage: q q/pqload.q -root ~/.cache/market-warehouse/ohlcv -mkts US IN -from 2018.01.01 -to 2020.12.31 -names 150 -hdb db

o:.Q.opt .z.x
if[not all `root`mkts`from`to`names`hdb in key o; -2 "usage: q q/pqload.q -root DIR -mkts M1 [M2..] -from D -to D -names N -hdb DIR"; exit 1]
qdir:$[1<count p:"/" vs string .z.f; "/" sv -1_p; "."]
system "l ",qdir,"/schema.q"
([pq]):use`kx.pq

etf:("*BEES*";"*ETF*";"*LIQUID*";"*GILT*";"*NIFTY*";"*SENSEX*")   / upstream India ETF exclusion

/ one market: every year file overlapping the window, filtered, top n by liquidity
readmkt:{[root;d0;d1;n;mk]
  d:hsym `$root,"/",string mk;
  fs:f where (f:key d) like "year=*.parquet";
  fs:fs where ("J"$4#'5_'string fs) within `year$(d0;d1);
  t:raze {[d;d0;d1;f] x:select from pq ` sv d,f; select date:`date$Date, sym:`$Symbol, open:`float$Open, high:`float$High, low:`float$Low, close:`float$Close, volume:`float$Volume from x where (`date$Date) within (d0;d1), Close>0}[d;d0;d1] each fs;
  if[mk=`IN; t:select from t where not any (upper string sym) like/: etf];
  k:exec sym from n sublist `m xdesc 0!select m:med close*volume by sym from t where volume>0;
  t:0!select first open, first high, first low, first close, first volume by date, sym from t where sym in k;
  `date`mkt`sym`open`high`low`close`volume#update mkt:mk from t}

root:{$["~"=first x; getenv[`HOME],1_x; x]} first o`root
d0:"D"$first o`from; d1:"D"$first o`to; n:"J"$first o`names
t:raze readmkt[root;d0;d1;n] each `$o`mkts
writehdb[hsym `$first o`hdb; t];
-1 "pq: ",string[count t]," rows, ",(", " sv {string[x],": ",string y}'[key c;value c:count each group t`mkt])," -> ",first o`hdb;
exit 0
