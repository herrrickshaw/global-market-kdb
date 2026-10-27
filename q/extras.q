/ extras.q - additions beyond the upstream study, built on the same gated panel.
/   .ext.legsic    rank IC of each signal against the overnight and intraday
/                  legs of its forward return (daily bars: close->open, open->close)
/   .ext.tca       quintile spreads net of a per-name Corwin-Schultz spread cost,
/                  next to upstream's flat per-market cost
/   .ext.fund      annual fundamentals read with pq, available `lag` days after FY end
/   .ext.ep        earnings yield joined point-in-time with aj (no look-ahead)
/   .ext.pitic     rank IC of the earnings yield
/   .ext.cafile    split/bonus factors read with pq
/   .ext.adjust    back-adjust bars for splits and bonuses
/   .ext.caeffect  the study on raw vs adjusted bars: guard hits and IC
/ .
/ Entry point: .ext.run[t;n;start;fund;ca] -> dictionary of tables
/   fund, ca: tables from .ext.fund / .ext.cafile, or () to skip

\d .ext

([pq]):use`kx.pq
sigs:.sig.sigs
M:{[x;c] (count distinct x`sym) cut x c}

/ ---- overnight vs intraday ---------------------------------------------------

legsic:{[x;mk]
  S:sigs!M[x] each sigs; L:`ol5`il5`ol21`il21!M[x] each `ol5`il5`ol21`il21;
  raze {[mk;S;L;k] {[mk;S;L;k;c]
      h:"J"$2_string c; ic:.study.icser[S k;L c]; a:.st.nwt[ic;.study.lags h];
      `market`signal`h`leg`mean_ic`nw_t`n_dates!(mk;k;h;$["o"=first string c;`overnight;`intraday];a 0;a 1;count .st.nn ic)}[mk;S;L;k] each key L}[mk;S;L] each sigs}

/ largest |overnight + intraday - log(1 + fwd)| where both legs exist: 0 up to rounding
legcheck:{[x] max {[x;h] l:x[`$"ol",string h]+x[`$"il",string h]; f:log 1+x[`$"f",string h]; k:where (not null l)&not null f; max abs l[k]-f k}[x] each 5 21}

/ ---- transaction costs from a per-name spread estimate -----------------------

/ quintile book on a non-overlapping grid; each rebalance pays half the spread
/ on every name bought or sold, as a fraction of the leg (spread missing -> the
/ date's median)
qtca:{[S;F;C]
  ls:0#0f; lo:0#0f; cl:0#0f; cg:0#0f; ms:0#0f; pt:0#0; pb:0#0; have:0b; i:0;
  while[i<count S;
    s:S i; f:F i; c:C i; ok:where (not null s)&not null f;
    md:med c where not null c; c:md^c;
    $[count[ok]<50; have:0b;
      [r:.study.prank s ok; top:ok where r>0.8; bot:ok where r<=0.2;
       ls,:(avg f top)-avg f bot; lo,:(avg f top)-avg f ok; ms,:md;
       if[have;
         ct:0.5*((sum c top except pt)+sum c pt except top)%count top;
         cb:0.5*((sum c bot except pb)+sum c pb except bot)%count bot;
         cl,:ct+cb; cg,:ct];
       pt:top; pb:bot; have:0<count top]];
    i+:1];
  `ls`lo`cost_ls`cost_long`spread!(avg ls;avg lo;$[count cl;avg cl;0n];$[count cg;avg cg;0n];med ms)}

tca:{[x;mk]
  D:count distinct x`date; S:sigs!M[x] each sigs; F:.sig.H!M[x] each `f5`f10`f21`f63; C:M[x]`cs;
  raze {[mk;D;S;F;C;h] {[mk;D;S;F;C;h;k]
      st:.study.lags h; sg:$[(avg .study.icser[S k;F h])>=0; 1f; -1f]; ix:where 0=(til D) mod st;
      q:qtca[sg*S[k] ix;F[h] ix;C ix]; ppy:252%h;
      `market`signal`h`direction`median_spread_bps`ls_cost_bps`ls_net_ann_pct_tca`long_net_ann_pct_tca!(mk;k;h;$[sg>0;`$"+";`$"-"];1e4*q`spread;1e4*q`cost_ls;(100*q[`ls]-q`cost_ls)*ppy;(100*q[`lo]-q`cost_long)*ppy)}[mk;D;S;F;C;h] each sigs}[mk;D;S;F;C] each 5 21 63}

/ ---- point-in-time fundamentals ----------------------------------------------

/ annual records, usable from fy_end + lag; one row per (sym, availability date)
fund:{[file;lag]
  f:select sym:`$ticker, fy:fy_end, date:lag+fy_end, ni:net_income, shares from pq file;
  `sym`date xasc 0!select last fy, last ni, last shares by sym, date from f}

/ earnings yield on each panel row from the latest record available by that date
ep:{[x;f]
  j:aj[`sym`date;select date,sym,close from x;select sym,date,fy,ni,shares,avail:date from f];
  update ep:?[(shares>0)&close>0; ni%shares*close; 0n] from j}

pitic:{[x;f;mk;lag]
  j:ep[x;f]; x:update ep:j`ep from x;
  la:exec all avail<=date from j where not null avail;                 / no look-ahead
  E:M[x]`ep; F:.sig.H!M[x] each `f5`f10`f21`f63;
  cv:(sum not null x`ep)%sum not null x`close;
  {[mk;E;F;cv;la;lag;h] ic:.study.icser[E;F h]; a:.st.nwt[ic;.study.lags h];
    `market`signal`h`lag_days`mean_ic`nw_t`p`n_dates`coverage`no_lookahead!(mk;`ep;h;lag;a 0;a 1;a 2;count .st.nn ic;cv;la)}[mk;E;F;cv;la;lag] each .sig.H}

/ ---- corporate actions -------------------------------------------------------

/ split/bonus factors: prices before ex-date are multiplied by factor (0.5 = 1:1 bonus)
cafile:{[file]
  c:distinct select sym:`$symbol, date:`date$ex_date, kind:`$kind, factor from pq file;
  `sym`date xasc 0!select factor:prd factor by sym, date from c}

/ multiplier for each bar date d given ascending ex-dates e and factors f
mult:{[d;e;f] ((reverse prds reverse f),1f) 1+e bin d}

adjust:{[t;ca]
  ev:exec sym!flip (e;f) from 0!select e:date, f:factor by sym from ca;
  t:update m:{[ev;s;d] $[s in key ev; .ext.mult[d;ev[s] 0;ev[s] 1]; count[d]#1f]}[ev;first sym;date] by sym from `sym`date xasc t;
  `date xasc delete m from update open:open*m, high:high*m, low:low*m, close:close*m, volume:volume%m from t}

caeffect:{[t;ca;n;start;mk]
  raw:select date,sym,open,high,low,close,volume from t where mkt=mk;
  adj:adjust[raw;ca];
  hits:{[t] sum {r:-1+x%prev x; sum (not null r)&(r<-0.5)|r>1} each value exec close by sym from `sym`date xasc t};
  xr:.sig.build[raw;n;start]; xa:.sig.build[adj;n;start];
  ics:{[x;k;h] .study.icser[M[x] k;M[x] `$"f",string h]};
  rows:raze {[mk;xr;xa;ics;k] {[mk;xr;xa;ics;k;h] r:.st.nwt[ics[xr;k;h];.study.lags h]; a:.st.nwt[ics[xa;k;h];.study.lags h];
      `market`signal`h`mean_ic_raw`nw_t_raw`mean_ic_adj`nw_t_adj`obs_raw`obs_adj!(mk;k;h;r 0;r 1;a 0;a 1;sum not null xr k;sum not null xa k)}[mk;xr;xa;ics;k] each 5 21}[mk;xr;xa;ics] each sigs;
  g:`market`events_applied`guard_hits_raw`guard_hits_adj!(mk;count select from ca where sym in distinct raw`sym;sum hits raw;sum hits adj);
  `corpact_effect`corpact_guard!(rows;enlist g)}

/ ---- entry point -------------------------------------------------------------

run:{[t;n;start;fund;ca]
  mks:distinct t`mkt;
  xs:mks!{[t;n;start;mk] .sig.build[select date,sym,open,high,low,close,volume from t where mkt=mk;n;start]}[t;n;start] each mks;
  out:`overnight_intraday`costs_tca`legs_check!(raze legsic'[value xs;mks]; raze tca'[value xs;mks]; ([] market:mks; max_abs_dev:legcheck each value xs));
  if[(`IN in mks)&98h=type fund; out[`pit_ep]:pitic[xs`IN;fund;`IN;first exec date-fy from fund]];
  if[(`IN in mks)&98h=type ca; out,:caeffect[t;ca;n;start;`IN]];
  out}

\d .
