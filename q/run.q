/ run.q - load an HDB, build the signals for one market, write per-date rank IC.
/ usage: q q/run.q -hdb db -mkt SYN -n 100 -start 2019.01.01 -out out/ic_q.csv

o:.Q.opt .z.x
if[not all `hdb`mkt`n`start`out in key o; -2 "usage: q q/run.q -hdb DIR -mkt SYM -n N -start YYYY.MM.DD -out FILE"; exit 1]
qdir:$[1<count p:"/" vs string .z.f; "/" sv -1_p; "."]
system "l ",qdir,"/signals.q"

/ resolve paths before \l, which changes directory into the HDB
cwd:first system "cd"
absp:{$["/"=first x; x; cwd,"/",x]}
hdb:absp first o`hdb; out:absp first o`out
m:`$first o`mkt; n:"J"$first o`n; st:"D"$first o`start

system "l ",hdb
t:select date,sym,close,high,volume from daily where mkt=m
s:.sig.build[t;n;st]
r:.sig.ic s

system "P 0"                       / full float precision in the CSV
(hsym `$out) 0: "," 0: r
-1 "q: ",string[count t]," bars, ",string[count distinct s`date]," formation dates -> ",out;
show .sig.summary[r;ceiling .sig.H%5]
exit 0
