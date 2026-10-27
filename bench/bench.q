/ bench.q - the q side of bench/run.py. Times five operations the study does,
/ on the full Parquet warehouse, and prints one JSON line per operation.
/ usage: q bench/bench.q -root ~/.cache/market-warehouse -reps 5 [-s 4]

o:.Q.opt .z.x
root:{$["~"=first x; getenv[`HOME],1_x; x]} first o`root
reps:"J"$first o`reps
([pq]):use`kx.pq

/ median and min wall time in ms of `reps` runs of f . a, plus the last result
tm:{[f;a] r:(); v:(); do[reps; s:.z.p; v:f . a; r,:(`long$.z.p-s)%1e6]; (med r; v; min r)}
emit:{[op;res;chk] -1 .j.j `engine`op`ms`min_ms`check!(`q;op;res 0;res 2;chk);}
files:{[m] d:hsym `$root,"/ohlcv/",m; ` sv/: d,/:f where (f:key d) like "year=*.parquet"}

/ B1 load every US year file into one in-memory table
ld:{[dummy] raze {select date:`date$Date, sym:`$Symbol, close:`float$Close, volume:Volume from pq x} each files "US"}
r:tm[ld;enlist 0]; t:r 1; emit[`load_us;r;count t]
/ idiomatic: one year file per secondary thread
ldp:{[dummy] raze {select date:`date$Date, sym:`$Symbol, close:`float$Close, volume:Volume from pq x} peach files "US"}
r:tm[ldp;enlist 0]; emit[`load_us_peach;r;count r 1]

/ B2 median dollar volume per symbol and year (upstream universe pre-filter)
r:tm[{[t] select m:med close*volume by sym, y:date.year from t where close>0, volume>0};enlist t]
emit[`groupby_median;r;count r 1]
/ idiomatic: one year per secondary thread
r:tm[{[t] raze {[t;y] select m:med close*volume by sym, y:y from t where (date.year)=y, close>0, volume>0}[t] peach distinct `year$t`date};enlist t]
emit[`groupby_median_peach;r;count r 1]

/ B3 per-symbol time series: 1d return, 5d return, 63-row mean |r|
t:`sym`date xasc t
r:tm[{[t] update rev5:-1+close%5 xprev close, am:63 mavg abs -1+close%prev close by sym from t};enlist t]
emit[`window_by_sym;r;sum r[1]`am]

/ B4 cross-sectional rank of the 5d return on every date
x:r 1
r:tm[{[x] update rk:rank rev5 by date from x};enlist x]
emit[`rank_by_date;r;count r 1]
/ idiomatic: rows stored by date (as in a date-partitioned HDB), ranks per contiguous date block
xd:`date xasc x
r:tm[{[x] update rk:raze rank each (where differ x`date) cut x`rev5 from x};enlist xd]
emit[`rank_by_date_sorted;r;count r 1]

/ B5 as-of join: India daily bars to the latest annual fundamentals published 90 days after FY end
ind:`sym`date xasc raze {select date:`date$Date, sym:`$Symbol, close:`float$Close from pq x} each files "IN"
f:`sym`date xasc select sym:`$ticker, date:90+fy_end, ni:net_income, shares from pq hsym `$root,"/fundamentals_history/IN_screener_only_backup.parquet"
f:update `g#sym from f
r:tm[{[ind;f] aj[`sym`date;ind;f]};(ind;f)]
emit[`asof_join_in;r;sum not null r[1]`ni]
exit 0
