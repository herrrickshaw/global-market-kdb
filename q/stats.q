/ stats.q - the statistics the study needs, written from scratch in q.
/ Counterparts in python/upstream/signal_validation.py and scipy.stats:
/   nwt     newey_west_t (mean, HAC t-stat, two-sided Student-t p-value)
/   bh      bh_fdr (Benjamini-Hochberg q-values)
/   dsr     deflated_sharpe (Bailey & Lopez de Prado 2014)
/   tp      2*scipy.stats.t.sf(|t|, df), via the regularised incomplete beta
/   ncdf    scipy.stats.norm.cdf, via the regularised incomplete gamma
/   nppf    scipy.stats.norm.ppf (Acklam's rational start + two Halley steps)
/   ols     least squares through the normal equations
/ The special functions follow Numerical Recipes (betacf, gser, gcf) with a
/ Lanczos log-gamma; they agree with scipy to ~1e-14.

\d .st

pi:acos -1
emc:0.5772156649            / Euler-Mascheroni, to upstream's precision
eps:3e-16
tiny:1e-300
fix:{$[tiny>abs x; tiny; x]}
nn:{x where not null x}
/ sample standard deviation (ddof=1), two-pass; null for fewer than two values
sd1:{v:nn x; if[2>count v; :0n]; d:v-avg v; sqrt (sum d*d)%-1+count d}
var1:{v:nn x; if[2>count v; :0n]; d:v-avg v; (sum d*d)%-1+count d}

/ ---- special functions -----------------------------------------------------

lgc:0.99999999999980993 676.5203681218851 -1259.1392167224028 771.32342877765313 -176.61502916214059 12.507343278686905 -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7
lgamma:{[x] x-:1; a:lgc[0]+sum (1_lgc)%x+1+til 8; t:x+7.5; (0.5*log 2*pi)+((x+0.5)*log t)+(log a)-t}

/ continued fraction for the incomplete beta (modified Lentz)
betacf:{[a;b;x]
  qab:a+b; qap:a+1; qam:a-1;
  c:1f; d:1%fix 1-qab*x%qap; h:d; m:1;
  while[m<=300;
    m2:2*m;
    aa:m*(b-m)*x%(qam+m2)*a+m2;
    d:1%fix 1+aa*d; c:fix 1+aa%c; h*:d*c;
    aa:neg (a+m)*(qab+m)*x%(a+m2)*qap+m2;
    d:1%fix 1+aa*d; c:fix 1+aa%c; del:d*c; h*:del;
    $[eps>abs del-1; m:301; m+:1]];
  h}

/ regularised incomplete beta I_x(a,b)
betai:{[a;b;x]
  if[x<=0; :0f]; if[x>=1; :1f];
  bt:exp (lgamma[a+b]-lgamma[a]+lgamma b)+(a*log x)+b*log 1-x;
  $[x<(a+1)%a+b+2; bt*betacf[a;b;x]%a; 1-bt*betacf[b;a;1-x]%b]}

/ two-sided Student-t p-value for statistic t with df degrees of freedom
tp:{[t;df] $[null t; 0n; betai[0.5*df;0.5;df%df+t*t]]}

/ regularised upper incomplete gamma Q(a,x): series below a+1, continued fraction above
gser:{[a;x] ap:a; s:1%a; del:s; i:0; while[i<1000; ap+:1; del*:x%ap; s+:del; $[(abs del)<eps*abs s; i:1000; i+:1]]; s*exp (neg x)+(a*log x)-lgamma a}
gcf:{[a;x] b:x+1-a; c:1%tiny; d:1%b; h:d; i:1; while[i<1000; an:neg i*i-a; b+:2; d:1%fix b+an*d; c:fix b+an%c; del:d*c; h*:del; $[eps>abs del-1; i:1000; i+:1]]; h*exp (neg x)+(a*log x)-lgamma a}
gq:{[a;x] $[x<=0; 1f; x<a+1; 1-gser[a;x]; gcf[a;x]]}

/ standard normal CDF: Phi(z) = Q(1/2, z^2/2)/2 for z < 0
ncdf:{[z] $[null z; 0n; z<0; 0.5*gq[0.5;0.5*z*z]; 1-0.5*gq[0.5;0.5*z*z]]}

/ standard normal quantile
pa:-3.969683028665376e+01 2.209460984245205e+02 -2.759285104469687e+02 1.383577518672690e+02 -3.066479806614716e+01 2.506628277459239e+00
pb:-5.447609879822406e+01 1.615858368580409e+02 -1.556989798598866e+02 6.680131188771972e+01 -1.328068155288572e+01 1
pc:-7.784894002430293e-03 -3.223964580411365e-01 -2.400758277161838e+00 -2.549732539343734e+00 4.374664141464968e+00 2.938163982698783e+00
pd:7.784695709041462e-03 3.224671290700398e-01 2.445134137142996e+00 3.754408661907416e+00 1
horner:{[c;x] {[x;a;k] k+a*x}[x]/[0f;c]}
nppf:{[p]
  x:$[p<0.02425; [q:sqrt -2*log p; horner[pc;q]%horner[pd;q]];
      p>1-0.02425; [q:sqrt -2*log 1-p; neg horner[pc;q]%horner[pd;q]];
      [q:p-0.5; r:q*q; q*horner[pa;r]%horner[pb;r]]];
  do[2; e:ncdf[x]-p; u:e*(sqrt 2*pi)*exp 0.5*x*x; x-:u%1+0.5*x*u];
  x}

/ ---- estimators ------------------------------------------------------------

/ mean, Newey-West (Bartlett) HAC t-stat with L lags, two-sided p (df = n-1)
nwt:{[x;L]
  v:nn x; n:count v;
  if[n<3; :3#0n];
  mu:avg v; e:v-mu; vv:(e wsum e)%n;
  vv+:sum {[e;n;L;k] 2*(1-k%L+1)*((k _ e) wsum neg[k] _ e)%n}[e;n;L] each 1+til L&n-1;
  t:mu%sqrt (tiny|vv)%n;
  (mu;t;tp[abs t;n-1])}

/ Benjamini-Hochberg q-values, nulls preserved
bh:{[p]
  q:count[p]#0n; ok:where not null p; m:count ok;
  if[m=0; :q];
  pv:p ok; o:iasc pv;
  r:reverse mins reverse pv[o]*m%1+til m;
  @[q;ok;:;@[m#0n;o;:;0f|1f&r]]}

/ least squares beta for y on the rows of X, where X is a list of k column vectors
ols:{[X;y] (inv X mmu flip X) mmu X mmu y}

/ Deflated Sharpe probability of per-period returns r, given n trials and the
/ variance of Sharpe ratios across those trials
dsr:{[r;ntr;v]
  r:nn r; T:count r;
  if[T<10; :0n];
  s:sd1 r; if[s=0; :0n];
  mu:avg r; sr:mu%s;
  e:r-mu; m2:avg e*e; g3:(avg e*e*e)%m2 xexp 1.5; g4:(avg e*e*e*e)%m2*m2;
  n:2|ntr; sd:sqrt 0f|v;
  sr0:sd*((1-emc)*nppf 1-1%n)+emc*nppf 1-1%n*exp 1;
  den:sqrt 1e-12|1+(neg g3*sr)+0.25*(g4-1)*sr*sr;
  ncdf (sr-sr0)*(sqrt T-1)%den}

\d .
