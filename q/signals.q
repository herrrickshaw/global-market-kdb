/ signals.q - the upstream signal panel in q: twelve signals, forward returns at
/ four horizons, gating exposures and the trailing liquidity gate.
/ A port of build_signals in python/upstream/signal_research.py. Every window,
/ min-periods rule and null convention follows pandas, so the two agree to
/ float rounding.
/ .
/ Entry point:
/   .sig.build[t;n;start]  t: table date,sym,open,high,low,close,volume (one market)
/                          n: universe size for the liquidity gate
/                          start: first formation date
/                          -> one row per (formation date, sym), every name on
/                             every date, values null outside the gate

\d .sig

sigs:`rev5`rsi_os`mom21`mom126`mom12_1`trend`golden`hi52`lowvol`illiq`volsurge   / upstream order
H:5 10 21 63             / forward horizons, trading days
minnames:30              / fewer valid names on a date -> null IC

/ ---- pandas-compatible building blocks -------------------------------------

/ ffill(limit=n): carry the last value forward over at most n nulls
ffill:{[n;x] i:maxs ?[null x;-1;til count x]; ?[(i>=0)&n>=(til count x)-i; x i; 0n]}

/ pct_change(fill_method=None)
pct:{-1+x%prev x}

/ ewm(alpha=a, adjust=False, min_periods=minp).mean() with nulls in the input.
/ State is (mean; weight of the old mean). A null input decays the old weight
/ but keeps the mean, exactly as pandas does with ignore_na=False.
ewstep:{[a;s;c]
  w:s 0; ow:s 1;
  if[null w; :$[null c; s; (c;1f)]];
  ow*:1-a;
  if[null c; :(w;ow)];
  if[w=c; :(w;1f)];
  (((ow*w)+a*c)%ow+a;1f)}
ewm:{[a;minp;x] r:ewstep[a]\[(0n;1f);x]; ?[(1|minp)<=sums not null x; r[;0]; 0n]}

/ rolling(n, min_periods=m): null unless the window holds >= m non-null values
rcnt:{[n;x] n msum not null x}
rmax:{[n;m;x] ?[m<=rcnt[n;x]; n mmax x; 0n]}
rmean:{[n;m;x] c:rcnt[n;x]; ?[m<=c; (n msum 0f^x)%c; 0n]}
rstd:{[n;m;x] c:rcnt[n;x]; s1:n msum 0f^x; s2:n msum 0f^x*x; ?[m<=c; sqrt 0f|(s2-s1*s1%c)%c-1; 0n]}
rmed:{[n;m;x] {[n;m;x;j] w:w where not null w:x j-til n; $[m<=count w; med w; 0n]}[n;m;x] each til count x}

/ clip(lower=0), keeping nulls null (0|0n is 0 in q, so guard explicitly)
clip0:{?[null x;0n;0f|x]}

/ log of the positive part: np.log(x.where(x > 0))
lnpos:{?[x>0;log x;0n]}

/ rank(method="average"), 1-based; nulls stay null
arank:{
  r:count[x]#0n; k:where not null x; v:x k;
  o:1+`float$iasc iasc v; g:value group v;
  @[r;k;:;@[o;g;:;avg each o g]]}

/ Wilder RSI-14 on simple returns
rsi:{[r1] up:ewm[1%14;14] clip0 r1; dn:ewm[1%14;14] clip0 neg r1; 100-100%1+up%?[dn>0;dn;0n]}

/ null x wherever the corporate-action flag fired in the trailing lb rows
clean:{[bad;lb;x] ?[0<lb mmax bad; 0n; x]}

/ h-day forward return entered at the next close, null if the window holds a flagged move
fwd:{[h;bad;c] ?[0<-1 xprev reverse (h+1) mmax reverse bad; 0n; -1+(neg[h+1] xprev c)%-1 xprev c]}

/ close above its 200-day mean (>= 150 obs), as 1/0, null where close is null
above:{[c] m:rmean[200;150;c]; ?[null c; 0n; `float$(not null m)&c>m]}

/ Corwin-Schultz (2012) high-low spread estimate from days t-1 and t; null on
/ missing or inconsistent bars, negative estimates set to zero
k3:3-2*sqrt 2
csd:{[h;l]
  ok:(h>0)&(l>0)&h>=l; ph:prev h; pl:prev l; ok&:(ph>0)&(pl>0)&ph>=pl;
  b:{x*x} (log h%l); b+:{x*x} log ph%pl;
  g:{x*x} log (h|ph)%l&pl;
  a:((sqrt 2*b)-sqrt b)%k3; a-:sqrt g%k3;
  ?[ok; 0f|2*(-1+exp a)%1+exp a; 0n]}

/ h-day overnight and intraday log legs of the same holding window as fwd
/ (close t+1 to close t+1+h): sum over j = t+2..t+1+h of log(O_j/C_j-1) and
/ log(C_j/O_j). Raw bars, so any missing day in the window gives null.
legs:{[h;bad;o;c]
  on:log o%prev c; id:log c%o;
  f:{[h;x] neg[h+1] xprev ?[h=h msum not null x; h msum 0f^x; 0n]}[h];   / complete windows only, then move to t
  m:0<-1 xprev reverse (h+1) mmax reverse bad;
  (?[m;0n;f on]; ?[m;0n;f id])}

/ last trading date of each Saturday..Friday week (2000.01.01 was a Saturday,
/ so d mod 7 is 6 on Fridays); matches pandas resample("W-FRI") sampling
weekends:{[d] d where w<>next w:d+6-(`int$d) mod 7}

/ ---- the panel ---------------------------------------------------------------

/ q-sql column expressions resolve names in the root namespace at run time,
/ so helpers are called fully qualified inside the updates below
build:{[t;n;start]
  d:asc distinct t`date; s:asc distinct t`sym;
  g:([] sym:raze (count d)#/:s; date:raze (count s)#enlist d);
  g:g lj `sym`date xkey select sym,date,open,high,low,close,volume from t;
  g:update rc:close, close:.sig.ffill[3] close by sym from g;
  g:update volume:0f^volume, r1:.sig.pct close by sym from g;
  g:update bad:`float$(not null r1)&(r1<-0.5)|r1>1, dv:close*volume from g;
  g:update e20:.sig.ewm[2%21;0;close], e50:.sig.ewm[2%51;0;close], e200:.sig.ewm[2%201;0;close] by sym from g;
  g:update rev5:.sig.clean[bad;6] neg -1+close%5 xprev close, rsi_os:.sig.clean[bad;30] neg .sig.rsi r1, mom21:.sig.clean[bad;22] -1+close%21 xprev close, mom126:.sig.clean[bad;127] -1+close%126 xprev close, mom12_1:.sig.clean[bad;253] -1+(21 xprev close)%252 xprev close by sym from g;
  g:update trend:.sig.clean[bad;60] -1+close%e50, golden:.sig.clean[bad;200] -1+e50%e200, hi52:.sig.clean[bad;253] close%.sig.rmax[252;200] high, lowvol:.sig.clean[bad;64] neg .sig.rstd[63;50] r1, illiq:.sig.clean[bad;64] .sig.rmean[63;50] abs[r1]%?[dv=0;0n;dv], volsurge:.sig.lnpos .sig.rmean[20;15;volume]%.sig.rmean[126;100;volume] by sym from g;
  g:update zone:?[null close;0n;?[close<e50;-1f;?[(close>e20)&e20>e50;1f;0f]]] from g;
  g:update cs:.sig.rmean[21;15] .sig.csd[high;low], ol5:first .sig.legs[5;bad;open;rc], il5:last .sig.legs[5;bad;open;rc], ol21:first .sig.legs[21;bad;open;rc], il21:last .sig.legs[21;bad;open;rc] by sym from g;
  g:update f5:.sig.fwd[5;bad;close], f10:.sig.fwd[10;bad;close], f21:.sig.fwd[21;bad;close], f63:.sig.fwd[63;bad;close], mdv:.sig.rmed[63;40;dv], vol:.sig.rstd[63;50] r1, ab200:.sig.above close, hist:252 msum not null close by sym from g;
  fd:weekends d; fd:fd where fd>=start;
  x:`date`sym xasc select from g where date in fd;
  / trailing liquidity gate: top n by 63d median dollar volume, >= 240 days of history
  x:update rk:.sig.arank neg mdv by date from x;
  x:update ok:(not null rk)&(rk<=n)&hist>=240 from x;
  c:sigs,`zone`f5`f10`f21`f63`mdv`vol`ab200`close`cs`ol5`il5`ol21`il21;
  x:![x;();0b;c!{(?;`ok;x;0n)} each c];
  (`date`sym,c)#x}

/ Spearman IC over names where both signal and forward return exist
ic1:{[x;f] k:where (not null x)&not null f; $[minnames>count k; 0n; cor[arank x k; arank f k]]}

\d .
