/ run.q - the full signal study from an HDB of daily bars, one CSV per output table.
/ usage: q q/run.q -hdb db -univ 100 -start 2019.01.01 -out out/q
/ Every market in the HDB is studied; BH-FDR and the Deflated Sharpe are taken
/ across all of them, as upstream does.

o:.Q.opt .z.x
if[not all `hdb`univ`start`out in key o; -2 "usage: q q/run.q -hdb DIR -univ N -start YYYY.MM.DD -out DIR"; exit 1]
qdir:$[1<count p:"/" vs string .z.f; "/" sv -1_p; "."]
{system "l ",qdir,"/",x,".q"} each ("stats";"signals";"study");

/ resolve paths before \l, which changes directory into the HDB
cwd:{$[10h=type x; x; first x]} system "cd"
absp:{$["/"=first x; x; cwd,"/",x]}
hdb:absp first o`hdb; out:absp first o`out
n:"J"$first o`univ; st:"D"$first o`start

system "l ",hdb
t:select date,mkt,sym,open,high,low,close,volume from daily
r:.study.run[t;n;st]

system "P 0"                       / full float precision in the CSVs
system "mkdir -p ",out;
{[out;r;k] (hsym `$out,"/",string[k],".csv") 0: "," 0: r k}[out;r] each key r;
-1 "q: ",string[count t]," bars, ",string[count distinct t`mkt]," market(s) -> ",out;
show select market,signal,h,mean_ic,nw_t,q_bh from r[`ic_table] where h=21
exit 0
