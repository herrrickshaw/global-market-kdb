/ study.q - the full upstream signal study in q.
/ A port of study_market() and the cross-market part of main() in
/ python/upstream/signal_research.py, using the panel from signals.q and the
/ estimators in stats.q. Per market:
/   ic table   mean rank IC, Newey-West t and p, annualised ICIR, hit rate,
/              liquidity/volatility-neutral IC, bull/bear regime IC
/              (12 signals x 4 horizons)
/   health     trailing-52-week IC vs full history, 10d horizon
/   fm         Fama-MacBeth joint coefficients, 10d horizon
/   costs      quintile spread, turnover, break-even, net-of-cost returns
/              on a non-overlapping grid (5/21/63d)
/   wf         purged expanding-window walk-forward: best single vs
/              IC-weighted vs ridge combination (10d)
/ then across markets: BH-FDR q-values, Harvey-Liu-Zhu |t| >= 3 flag, and the
/ Deflated Sharpe over every long-short strategy tried.
/ .
/ Entry point: .study.run[t;n;start] -> dictionary of tables
/   t: date,mkt,sym,open,high,low,close,volume (any number of markets)

\d .study

sigs:.sig.sigs
H:.sig.H
COST:`IN`US`JP`KR`EU!0.30 0.10 0.20 0.25 0.40    / round-trip %, upstream apply_costs.COSTS_PCT
nn:.st.nn
lags:{1|ceiling x%5}

/ per-date rank IC series between two lists of cross-section vectors
icser:{[S;F] .sig.ic1'[S;F]}

/ rank(pct=True) and the z-scored rank used by Fama-MacBeth and walk-forward
prank:{(.sig.arank x)%count nn x}
zrank:{r:prank x; v:nn r; (r-avg v)%.st.sd1 v}

/ residual of rank(signal) on [1, rank(mdv), rank(vol)] across names
neut1:{[y;a;b]
  y:prank y; a:prank a; b:prank b;
  ok:where (not null y)&(not null a)&not null b;
  if[count[ok]<13; :count[y]#0n];
  X:(count[ok]#1f;a ok;b ok);
  @[count[y]#0n;ok;:;(y ok)-.st.ols[X;y ok] mmu X]}

/ one Fama-MacBeth cross-section: slopes of z-ranked fwd on z-ranked signals
fm1:{[Z;y]
  k:count Z; ok:where (not null y)&all not null Z;
  if[count[ok]<50|k+10; :k#0n];
  1_.st.ols[(enlist count[ok]#1f),Z[;ok];y ok]}

/ quintile spread and turnover on a non-overlapping grid (quantile_spread_turnover)
qst:{[S;F]
  ls:0#0f; lo:0#0f; tl:0#0f; ts:0#0f; pt:0#0; pb:0#0; have:0b; i:0;
  while[i<count S;
    s:S i; f:F i; ok:where (not null s)&not null f;
    $[count[ok]<50; have:0b;
      [r:prank s ok; top:ok where r>0.8; bot:ok where r<=0.2;
       ls,:(avg f top)-avg f bot; lo,:(avg f top)-avg f ok;
       if[have; tl,:1-(count top inter pt)%1|count top; ts,:1-(count bot inter pb)%1|count bot];
       pt:top; pb:bot; have:0<count top]];
    i+:1];
  taul:$[count tl; avg tl; 0n]; taus:$[count ts; avg ts; 0n];
  tau:$[count v:nn (taul;taus); avg v; 0n];
  spread:$[count ls; avg nn ls; 0n]; excess:$[count lo; avg nn lo; 0n];
  `n_periods`ls_spread`ls_sharpe`long_excess`turnover_long`turnover_ls`breakeven_rt_cost_ls`ls_series!(count ls;spread;$[2<count ls;(avg nn ls)%.st.sd1 ls;0n];excess;taul;tau;$[(not null tau)&tau>0;spread%2*tau;0n];ls)}

/ health status of a 10d IC series (ic_health)
sgn:{$[null x;0n;x>0;1f;x<0;-1f;0f]}
health1:{[ic]
  f:.st.nwt[ic;2]; v:nn ic; r:.st.nwt[(neg 52&count v)#v;2];
  st:$[(null f 1)|2>abs f 1; `NOISE;
       (not sgn[r 0]=sgn f 0)&2<=abs r 1; `FLIPPED;
       (sgn[r 0]=sgn f 0)&(abs r 0)>=0.5*abs f 0; `HEALTHY;
       `DECAYING];
  `full_ic`full_t`recent_ic`recent_t`status!(f 0;f 1;r 0;r 1;st)}

/ purged walk-forward (purged_walk_forward): Z dict signal -> z-ranked panel,
/ Y z-ranked fwd, F raw fwd, dts formation dates, emb embargo in dates
wf:{[mk;Z;Y;F;dts;emb]
  n:count dts;
  if[(n-104)<20; :()];
  e:floor 104+((n-104)%5)*til 6; e[5]:n;
  fold:{[mk;Z;Y;F;dts;emb;e;i]
    t0:e i; t1:e i+1; tr:til 0|t0-emb; te:t0+til t1-t0;
    if[52>count tr; :()];
    ict:{[F;tr;z] avg nn icser[z tr;F tr]}[F;tr] each Z sigs;
    a:{$[null x;-1f;abs x]} each ict; b:first where a=max a; best:sigs b; sg:sgn ict b;
    X:{[tr;z] raze z tr}[tr] each Z sigs; y:raze Y tr;
    ok:where (not null y)&all not null X; X:X[;ok]; y:y ok; nobs:count y;
    beta:(inv ((X mmu flip X)%nobs)+0.1*{x=/:x} til count sigs) mmu (X mmu y)%nobs;
    w:0f^ict;
    pb:{[Z;best;sg;j] sg*Z[best] j}[Z;best;sg] each te;
    pw:{[Z;w;j] sum w*Z[sigs][;j]}[Z;w] each te;
    pr:{[Z;beta;j] sum beta*Z[sigs][;j]}[Z;beta] each te;
    oos:{[F;te;p] avg nn icser[p;F te]}[F;te] each (pb;pw;pr);
    enlist `market`fold`test_start`test_end`best_single_name`oos_ic_best_single`oos_ic_ic_weighted`oos_ic_ridge!(mk;i+1;dts t0;dts t1-1;best;oos 0;oos 1;oos 2)}[mk;Z;Y;F;dts;emb;e];
  raze fold each til 5}

/ one market's study: x is the gated panel from .sig.build
market:{[x;mk]
  ns:count distinct x`sym; dts:distinct x`date; D:count dts;
  M:{[ns;x;c] ns cut x c}[ns;x];
  S:(sigs,`zone)!M each sigs,`zone; F:H!M each `f5`f10`f21`f63;
  br:avg each M`ab200; bear:(not null br)&br<0.45;
  ne:sigs!{[x;s] neut1'[s;x`mdv;x`vol]}[`mdv`vol!(M`mdv;M`vol)] each S sigs;
  c:`mk`S`F`ne`bear!(mk;S;F;ne;bear);
  ic:raze {[c;k] {[c;k;h]
      L:lags h; ic:icser[c[`S] k;c[`F] h]; a:.st.nwt[ic;L]; sd:.st.sd1 ic;
      nb:$[k in key c`ne; 2#.st.nwt[icser[c[`ne] k;c[`F] h];L]; 0n 0n];
      bu:2#.st.nwt[ic where not c`bear;L]; be:2#.st.nwt[ic where c`bear;L];
      `market`signal`h`mean_ic`nw_t`p`icir_ann`hit`n_dates`neutral_ic`neutral_t`bull_ic`bull_t`bear_ic`bear_t!(c`mk;k;h;a 0;a 1;a 2;$[(null sd)|sd=0f;0n;(a[0]%sd)*sqrt 52%L];avg ic>0;count nn ic),nb,bu,be}[c;k] each H}[c] each sigs,`zone;
  series:raze {[c;dts;k] raze {[c;dts;k;h] ([] market:count[dts]#c`mk; signal:count[dts]#k; h:count[dts]#h; date:dts; ic:icser[c[`S] k;c[`F] h])}[c;dts;k] each H}[c;dts] each sigs,`zone;
  hl:{[c;k] (`market`signal!(c`mk;k)),health1 icser[c[`S] k;c[`F] 10]}[c] each sigs;
  Z:sigs!{zrank each x} each S sigs; Y:zrank each F 10;
  C:{[Z;Y;j] fm1[Z[sigs][;j];Y j]}[Z;Y] each til D;
  fm:{[mk;C;i] a:.st.nwt[C[;i];2]; `market`signal`fm_coef`fm_t`fm_p`n_dates!(mk;sigs i;a 0;a 1;a 2;count nn C[;i])}[mk;C] each til count sigs;
  cr:$[mk in key COST; COST mk; 0.3]%100;
  cost:raze {[c;D;cr;h] {[c;D;cr;h;k]
      st:lags h; sg:$[(avg icser[c[`S] k;c[`F] h])>=0; 1f; -1f]; ix:where 0=(til D) mod st;
      q:qst[sg*c[`S][k] ix;c[`F][h] ix]; ppy:252%h;
      lsn:q[`ls_spread]-2*q[`turnover_ls]*cr; lon:q[`long_excess]-q[`turnover_long]*cr;
      `market`signal`h`direction`ls_spread_pct`turnover_ls`breakeven_bps_ls`cost_bps`ls_net_ann_pct`long_excess_pct`turnover_long`long_net_ann_pct`ls_sharpe_ann`n_periods`ls_series!(c`mk;k;h;$[sg>0;`$"+";`$"-"];100*q`ls_spread;q`turnover_ls;1e4*q`breakeven_rt_cost_ls;1e4*cr;(100*lsn)*ppy;100*q`long_excess;q`turnover_long;(100*lon)*ppy;q[`ls_sharpe]*sqrt ppy;q`n_periods;q[`ls_series]-2*q[`turnover_ls]*cr)}[c;D;cr;h] each sigs}[c;D;cr] each 5 21 63;
  w:wf[mk;Z;Y;F 10;dts;1+lags 10];
  `ic_series`ic_table`fama_macbeth_10d`walk_forward_10d`costs`health!(series;ic;fm;w;cost;hl)}

/ every market in t, then the cross-market post-processing of upstream main()
run:{[t;n;start]
  r:{[t;n;start;mk] market[.sig.build[select date,sym,open,high,low,close,volume from t where mkt=mk;n;start];mk]}[t;n;start] each distinct t`mkt;
  out:(key r 0)!{[r;k] raze r@\:k}[r] each key r 0;
  ic:out`ic_table; ic:update q_bh:.st.bh p, hlz_t3:3<=abs nw_t from ic; out[`ic_table]:ic;
  cost:out`costs; ser:cost`ls_series;
  srs:{(avg nn x)%.st.sd1 x} each ser where 10<count each ser;
  v:.st.var1 srs;
  ntr:count cost;
  out[`costs]:delete ls_series from update dsr:{[ntr;v;s] $[10<count s; .st.dsr[s;ntr;v]; 0n]}[ntr;v] each ls_series from cost;
  out}

\d .
