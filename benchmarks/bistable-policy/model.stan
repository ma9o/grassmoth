// Lean centered version: avoid saving a second copy of every latent state.
functions {
  vector means_x(vector x, vector z, real a) {
    return x + 0.2 * (1.15 * (1 - a * z) .* (1 - 0.35 * inv_logit(4 * (x - 5))) - 1)
      + 0.1 * inv_logit(4 * (1 - x)) - 1e-6 * x;
  }
  vector means_z(vector x, vector z, vector u, real h) {
    return z + (inv_logit(16 * (z + 0.05 * (x - 4) + u - h)) - z) / 7;
  }
}
data {
  int<lower=2> T;
  int<lower=1> N;
  array[N] int<lower=1,upper=T> observed;
  vector[N] y;
  vector[T] u;
}
parameters {
  vector[2] theta_std;
  vector[T] x;
  vector[T] z;
}
transformed parameters {
  real a = 0.2 + 0.15 * theta_std[1];
  real h = 0.5 + 0.05 * theta_std[2];
}
model {
  theta_std ~ std_normal();
  x[1] ~ normal(1.5, 0.5);
  z[1] ~ normal(0, 0.1);
  tail(x, T - 1) ~ normal(means_x(head(x, T - 1), head(z, T - 1), a), 0.08);
  tail(z, T - 1) ~ normal(means_z(head(x, T - 1), head(z, T - 1), head(u, T - 1), h), 0.03);
  y ~ normal(x[observed], 0.2);
}
