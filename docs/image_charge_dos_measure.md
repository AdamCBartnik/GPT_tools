**Semiconductor DOS and the population-flux emission model**

The population-flux model derived here is now the sole semiconductor model in
`GPT_tools/image_charge.py`. Omitted `effective_mass` defaults to 1.0;
transverse momentum is always conserved. Explicit `None`, zero, negative,
and nonfinite mass ratios are invalid. Earlier runs intentionally change.
Comparisons to the former model below are retained as derivation history.

**Conclusion.** The current semiconductor energy weight does not follow from
an isotropic excited population with the stated DOS product and an
energy-independent conversion from excitation rate to population. It contains
an extra factor of the internal momentum, proportional to the square root of
the conduction-band kinetic energy. Removing that factor is the appropriate
correction for a *population-flux* model. Counting eventual escape from a
finite pulse-created population is a different transport model and need not
give the same answer. Excitation assumptions alone do not choose between them.

**1. Keep the three measures separate.**

Write internal kinetic energy as q = K + chi, with Delta = photon energy minus
band gap, r = m*/m_e, and mu = cos(theta_i). Assume isotropic parabolic bands,
occupied valence states, empty final states, a constant transition matrix
element, and photoexcitation without a crystal-momentum selection rule.

The excitation rate per unit internal energy and solid angle is then

    S(q) dq dOmega proportional to sqrt(Delta-q) sqrt(q) dq dOmega,
    0 < q < Delta.

This includes the final-state DOS once. The excitation rate *per final state*
instead contains only sqrt(Delta-q); summing that rate over final states
produces the extra sqrt(q). Fermi's golden rule distinguishes a rate into one
state from a rate summed over a continuum. Momentum-conserving direct optical
transitions require a joint band calculation instead of this DOS product;
the present proposal retains the notes' momentum-randomized excitation
assumption. [MIT optical-properties notes, Appendix A and Chapters 4–5](https://web.mit.edu/course/6/6.732/www/opt.pdf)

If the population n(q) per unit energy is proportional to S(q), its outward
crossing flux is

    dJ proportional to n(q) v(q) mu dq dmu,
    v(q) = sqrt(2q/m*).

Consequently the hemisphere-integrated internal flux is proportional to
q sqrt(Delta-q), and its normalized angular measure is 2 mu dmu. Equivalently,
sin^2(theta_i) is uniform.

Two independent routes expose the extra factor in the current calculation:

* At fixed q, dOmega = 2 pi dp_z / p_i. Multiplying by v_z = p_z/m*
  gives n(q) p_z dp_z / (m* p_i). The notes omit the energy-dependent 1/p_i.
* In phase space, the excited occupation per state is proportional to
  sqrt(Delta-q). The cylindrical measure is d^3p = 2 pi m* dq dp_z.
  Multiplication by v_z therefore leaves sqrt(Delta-q) p_z dq dp_z.
  Inserting another final-state DOS here counts its energy dependence twice.

An energy-dependent effective population lifetime changes the result:
n(q) proportional to S(q) tau(q). The current weight can be reproduced by
tau(q) proportional to sqrt(q), up to a dimensional reference constant. This
would be an additional transport assumption, not a consequence of the DOS.

**2. The proposed population-flux weight.**

Classical transverse-momentum conservation is unchanged. A trajectory reaches
the specified barrier potential when

    r q (1-mu^2) <= K = q-chi.

The fraction of internal flux directions admitted is

    A(q,chi) = min(1, (q-chi)/(r q)),

on q in [max(0,chi), Delta]. Applying A to the internal flux gives

    W_population(K) proportional to min(K+chi, K/r) sqrt(Eexc-K),
    Eexc = Delta-chi.

Compare with the current expression:

    W_legacy(K) proportional to min(K+chi, K/r)
                               sqrt(K+chi) sqrt(Eexc-K).

The factor sqrt(K+chi) is the only difference in this mode. It is not removed
by normalizing a distribution because it varies across the spectrum.

Given K, the current outgoing sampler is still correct for population flux:

    sin^2(theta_o) uniform on [0, min(1, r*(K+chi)/K)],
    mean(K_perp | K) = min(K, r*(K+chi))/2.

Normalize W_population for the energy CDF; multiply it by the conditional MTE
above for the MTE numerator. Surface-to-peak transmission remains the ratio
of the peak and surface integrals of the *same* weight. This transmission is
conditional on surface emission and is not absolute electrons-per-photon QE.

The survivor-only construction remains valid for a static, planar external
potential. K decreases by the barrier height, q remains unchanged, and
transverse momentum remains unchanged. The peak ensemble can still be sampled
directly and have the longitudinal barrier energy restored at the surface.

**3. Why eventual escape counts are different.**

For an initially isotropic pulse-created source, every outward electron that
can escape contributes one count if there is no loss during transport. Faster
electrons arrive sooner, but do not count more. Multiplying the pulse-integrated
source by velocity again would be incorrect. In that limit mu is uniform,
rather than mu^2, and the angular acceptance is

    a = min(1, K/(r*q)),
    mu_min = sqrt(1-a),
    A_counts = 1-mu_min.

The energy weight and conditional MTE become

    W_counts(K) proportional to sqrt(q)*sqrt(Delta-q)*(1-mu_min),
    mean(K_perp | K) = r*q*(2-mu_min-mu_min^2)/3.

This is first-pass outward emission without scattering, return, or retries.
At fixed energy its sampler uses mu uniform on [mu_min,1] followed by refraction.
The current uniform-sin^2 angular sampler would not apply to this alternative.

An explicit transport model connects these cases. For an exponentially
distributed excitation depth with absorption length ell and a straight-path
loss length lambda(q), integrating the probability of reaching the surface
without a loss gives

    D(q,mu) = integral_0^infinity exp(-s/ell)/ell
                                * exp(-s/[lambda(q)*mu]) ds
            = lambda(q)*mu / [ell + lambda(q)*mu].

Thus source-to-surface counts are proportional to S(q) D(q,mu) dq dmu,
with no additional velocity multiplier. This is the same kind of depth and
path attenuation used in Dowell–Schmerge, whose Eq. (3) uses d(cos theta)
and a separate transport factor. Their near-threshold metal approximations
make internal speed and angular variations small; those approximations do
not establish an exact semiconductor flux law. [Dowell–Schmerge, Eqs. (3)–(7)](https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.12.074201)

For lambda much larger than ell, D approaches one and gives ballistic counts.
For lambda much smaller than ell, D is proportional to lambda(q)*mu.
In the latter limit, a constant loss time (lambda=v*tau) gives the proposed
population-flux shape. A constant mean free path gives a different energy
weight. This illustrates why a short optical pulse by itself does not settle
the angular or energy weighting: the transport and loss physics also matter.
Treating a collision as removal describes the unscattered component; it does
not model cooled electrons that can subsequently emit.

**4. Analytic checks and size of the change.**

At zero local affinity, for 0 < r <= 1, all internal directions transmit.
The following coefficients multiply r*Delta:

| Model | MTE/(r Delta) |
| --- | ---: |
| Current energy weight and flux angles | 5/16 = 0.312500 |
| Proposed isotropic population flux | 2/7 = 0.285714 |
| Lossless first-pass source counts | 1/3 = 0.333333 |

For population flux the zero-affinity energy distribution in q/Delta is
Beta(2,3/2), instead of the current Beta(5/2,3/2). The MTE change is -8.57%
relative to the current result. For r>1 the zero-affinity MTE is 2*Delta/7
because the internal angular acceptance is restricted.

A separate equilibrium sanity check is useful: an occupation per state
exp(-q/kT) gives outward flux q exp(-q/kT). When all internal angles transmit,
its MTE is r*kT. Retaining an extra sqrt(q) in that flux prescription would
give 5*r*kT/4. This check concerns a thermal population supplied to a generalized
kernel, not the current nonthermal valence-to-conduction excitation model.

Illustrative numerical comparison (not inferred from the user's runs):
Delta=0.3 eV, r=0.2, and surface chi=-0.1 eV. Each transmission is normalized
to its own model's surface integral; absolute unnormalized weights from
different models have different prefactors and must not be compared directly.

| Peak chi (eV) | Current MTE (meV) | Population-flux MTE (meV) | Current transmission | Population-flux transmission |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 18.7500 | 17.1429 | 1.000000 | 1.000000 |
| 0.10 | 20.4631 | 19.7698 | 0.852104 | 0.774871 |
| 0.20 | 22.5078 | 22.2181 | 0.343531 | 0.275389 |
| 0.29 | 2.86175 | 2.85714 | 0.00136766 | 0.00101430 |

The MTE difference can become small while the relative transmission difference
remains substantial. At peak chi=0.2 eV, this example changes transmission by
about -20% relative to the current value, while MTE changes by about -1.3%.
Ballistic counts at peak chi=0.1 eV instead give MTE=25.7371 meV and
transmission=0.643217, illustrating the separate effect of the transport choice.

**5. Implementation.**

No model selector is needed. The only interface parameter is the mass ratio:

    settings['effective_mass'] = 0.2

All semiconductor particle generation, CDF, PDF, inverse CDF, MTE, and
transmission routines use the same population-flux energy kernel. The default
mass ratio is 1.0, including when the setting is absent from the dictionary.
The existing angular and survivor algorithms remain. The previous DOS-weighted
integrals and no-refraction branches have been removed. Metal routines are
unchanged. Existing positional RNG and inverse-solver arguments retain their
positions; effective_mass remains keyword-only in those two APIs.

For the new integrals, put t=sqrt(Eexc-K)=sqrt(Delta-q). On either side of the
escape-cone crossing, define L(t)=min(Delta-t^2, (Eexc-t^2)/r). Then

    normalization integrand in t = 2*t^2*L(t),
    MTE numerator integrand in t = r*t^2*L(t)^2.

These are polynomials of degree four and six on each branch. Four-point
Gauss–Legendre integration on each interval evaluates both exactly up to
roundoff. Integrate definite intervals with positive weights and rationalize
differences of endpoint square roots to avoid the cancellation issue fixed
earlier. This method also supplies partial integrals for the CDF.

Do not implement ballistic_counts merely by changing the energy factor: it
also needs a different conditional angular sampler. Treat that as a later
explicit model if the intended source and transport assumptions warrant it.

The accompanying `image_charge_dos_comparison.py` is a scalar numerical
prototype of the population integral plus independent comparison tools.
It passed 84 comparisons of full/partial integrals and MTE against independent
quadrature, including both signs of affinity, r above and below one, and
near-threshold support. Exact zero-affinity limits were also checked. A
one-million-particle Monte Carlo using the internal DOS-product energy
distribution and uniform internal solid angle independently reproduced the
MTE and transmission for all three comparison models within 0.6% tolerance.

Production regression checks cover PDF/CDF/inverse consistency, the analytic
zero-affinity and restricted-cone limits, Monte Carlo surface filtering versus
direct survivor generation, default versus explicit unit mass, and positional
argument compatibility. They validate the measures and integrals; they do not
validate a material-specific transport model or constitute a full GPT simulation.
