/ ext_run.q - the extras (overnight/intraday IC, spread-based costs, point-in-time
/ earnings yield, corporate-action effect) from an HDB, one CSV per table.
/ usage: q q/ext_run.q -hdb db -univ 150 -start 2017.07.01 -out out/ext [-fund FILE [-lag 90]] [-ca FILE]

o:.Q.opt .z.x
if[not all `hdb`univ`start`out in key o; -2 "usage: q q/ext_run.q -hdb DIR -univ N -start YYYY.MM.DD -out DIR [-fund FILE [-lag DAYS]] [-ca FILE]"; exit 1]
qdir:$[1<count p:"/" vs string .z.f; "/" sv -1_p; "."]
{system "l ",qdir,"/",x,".q"} each ("stats";"signals";"study";"extras");

/ resolve paths before \l, which changes directory into the HDB
cwd:{$[10h=type x; x; first x]} system "cd"
absp:{$["~"=first x; getenv[`HOME],1_x; "/"=first x; x; cwd,"/",x]}
hdb:absp first o`hdb; out:absp first o`out
n:"J"$first o`univ; st:"D"$first o`start
fund:$[`fund in key o; .ext.fund[hsym `$absp first o`fund; $[`lag in key o; "J"$first o`lag; 90]]; ()]
ca:$[`ca in key o; .ext.cafile hsym `$absp first o`ca; ()]

system "l ",hdb
t:select date,mkt,sym,open,high,low,close,volume from daily
r:.ext.run[t;n;st;fund;ca]

system "P 0"
system "mkdir -p ",out;
{[out;r;k] (hsym `$out,"/",string[k],".csv") 0: "," 0: r k}[out;r] each key r;
-1 "ext: ",string[count t]," bars -> ",out,"  (",(", " sv string key r),")";
show r`legs_check
exit 0
