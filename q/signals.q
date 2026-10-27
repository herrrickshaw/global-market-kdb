/ signals.q - five equity signals, 21d forward return, liquidity gate and
/ cross-sectional rank IC. A port of build_signals / rank_ic_series /
/ newey_west_t in herrrickshaw/global-market-scanners (see python/reference.py
/ for the line-by-line Python counterpart). Every window, min-periods rule and
/ null convention follows pandas, so the two agree to float rounding.
/
/ Entry points:
/   .sig.build[t;n;start]  t: table date,sym,close,high,volume (one market)
/                          n: universe size for the liquidity gate
/                          start: first formation date
/                          -> one row per (formation date, sym) with the
/                             gated signals and fwd (21d forward return)
/   .sig.ic[s]             -> date,signal,ic,n   per-date Spearman rank IC
/   .sig.summary[r;L]      -> signal,mean_ic,nw_t,n_dates  (Newey-West t, L lags)

\d .sig

sigs:`rsi_os`hi52`rev5`golden`illiq
H:21                     / forward horizon, trading days
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
rmed:{[n;m;x] {[n;m;x;j] w:w where not null w:x j-til n; $[m<=count w; med w; 0n]}[n;m;x] each til count x}

/ clip(lower=0), keeping nulls null (0|0n is 0 in q, so guard explicitly)
clip0:{?[null x;0n;0f|x]}

/ rank(method="average"), 1-based; nulls stay null
arank:{
  r:count[x]#0n; k:where not null x; v:x k;
  o:1+`float$iasc iasc v; g:value group v;
  @[r;k;:;@[o;g;:;avg each o g]]}

/ Wilder RSI-14 on simple returns
rsi:{[r1] up:ewm[1%14;14] clip0 r1; dn:ewm[1%14;14] clip0 neg r1; 100-100%1+up%?[dn>0;dn;0n]}

/ null x wherever the corporate-action flag fired in the trailing lb rows
clean:{[bad;lb;x] ?[0<lb mmax bad; 0n; x]}

/ last trading date of each Saturday..Friday week (2000.01.01 was a Saturday,
/ so d mod 7 is 6 on Fridays); matches pandas resample("W-FRI") sampling
weekends:{[d] d where w<>next w:d+6-(`int$d) mod 7}

/ ---- signals and forward return on the full date x sym grid ----------------

build:{[t;n;start]
  d:asc distinct t`date; s:asc distinct t`sym;
  g:([] sym:raze (count d)#/:s; date:raze (count s)#enlist d);
  g:g lj `sym`date xkey select sym,date,close,high,volume from t;
  g:update close:ffill[3] close by sym from g;
  g:update volume:0f^volume, r1:pct close by sym from g;
  g:update bad:`float$(not null r1)&(r1<-0.5)|r1>1, dv:close*volume from g;
  g:update rsi_os:clean[bad;30] neg rsi r1, hi52:clean[bad;253] close%rmax[252;200] high, rev5:clean[bad;6] neg -1+close%5 xprev close by sym from g;
  g:update golden:clean[bad;200] -1+ewm[2%51;0;close]%ewm[2%201;0;close], illiq:clean[bad;64] rmean[63;50] abs[r1]%?[dv=0;0n;dv] by sym from g;
  g:update fwd:?[0<-1 xprev reverse (H+1) mmax reverse bad; 0n; -1+(neg[H+1] xprev close)%-1 xprev close], mdv:rmed[63;40;dv], hist:252 msum not null close by sym from g;
  fd:weekends d; fd:fd where fd>=start;
  x:`date`sym xasc select from g where date in fd;
  / trailing liquidity gate: top n by 63d median dollar volume, >= 240 days of history
  x:update rk:arank neg mdv by date from x;
  x:update ok:(not null rk)&(rk<=n)&hist>=240 from x;
  x:update rsi_os:?[ok;rsi_os;0n], hi52:?[ok;hi52;0n], rev5:?[ok;rev5;0n], golden:?[ok;golden;0n], illiq:?[ok;illiq;0n], fwd:?[ok;fwd;0n] from x;
  select date,sym,rsi_os,hi52,rev5,golden,illiq,fwd from x}

/ ---- evaluation ------------------------------------------------------------

/ Spearman IC over names where both signal and forward return exist
ic1:{[x;f] k:where (not null x)&not null f; $[minnames>count k; 0n; cor[arank x k; arank f k]]}

ic:{[s]
  g:group s`date; f:s`fwd;
  one:{[g;f;x;k] ([] date:key g; signal:(count g)#k; ic:{[x;f;j] ic1[x j;f j]}[x;f] each value g; n:{[x;f;j] sum (not null x j)&not null f j}[x;f] each value g)};
  raze {[one;s;g;f;k] one[g;f;s k;k]}[one;s;g;f] each sigs}

/ mean and HAC (Newey-West, Bartlett kernel) t-stat with L lags
nwt:{[x;L]
  v:x where not null x; n:count v;
  if[n<3; :0n 0n];
  mu:avg v; e:v-mu; vv:(e wsum e)%n;
  vv+:sum {[e;n;L;k] 2*(1-k%L+1)*((k _ e) wsum neg[k] _ e)%n}[e;n;L] each 1+til L&n-1;
  (mu; mu%sqrt (1e-300|vv)%n)}

summary:{[r;L]
  row:{[r;L;k] x:exec ic from r where signal=k; a:nwt[x;L]; (k;a 0;a 1;sum not null x)};
  flip `signal`mean_ic`nw_t`n_dates!flip row[r;L] each sigs}

\d .
