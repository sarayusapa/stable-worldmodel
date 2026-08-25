#set page(
  paper: "a4",
  margin: (x: 12mm, y: 10mm),
  fill: rgb("#f7f8fb"),
  footer: align(center, text(
    size: 7.3pt,
    fill: rgb("#5c687c"),
    [Design principle · use a reliable high-level prediction as the target for a constrained physical explanation.],
  )),
)
#set text(
  font: ("Libertinus Sans", "Noto Sans", "DejaVu Sans"),
  size: 8.4pt,
  fill: rgb("#172033"),
)
#set par(justify: true, leading: 0.55em)
#set math.equation(numbering: none)

#let navy = rgb("#172033")
#let blue = rgb("#3157d5")
#let cyan = rgb("#12a4a6")
#let amber = rgb("#d97706")
#let pale-blue = rgb("#eaf0ff")
#let pale-cyan = rgb("#e7f7f5")
#let pale-amber = rgb("#fff3df")
#let line = rgb("#dbe1ec")
#let muted = rgb("#5c687c")

#let section(title, body, accent: blue, fill: white) = block(
  width: 100%,
  fill: fill,
  stroke: (left: 2.2pt + accent, top: 0.5pt + line, right: 0.5pt + line, bottom: 0.5pt + line),
  radius: 4pt,
  inset: (x: 7pt, y: 6pt),
  [
    #text(size: 9.3pt, weight: "bold", fill: accent)[#title]
    #v(2.5pt)
    #body
  ],
)

#let tag(body, fill: pale-blue, ink: blue) = box(
  fill: fill,
  radius: 3pt,
  inset: (x: 5pt, y: 2pt),
  text(size: 7.2pt, weight: "bold", fill: ink, body),
)

#align(center)[
  #set par(justify: false)
  #text(size: 16.5pt, weight: "bold", fill: navy)[PhysWM: Top-Down Physical Representation Induction]
  #v(2pt)
  #text(size: 9.2pt, fill: muted)[Workshop-paper formulation and experiment contract]
]

#v(7pt)

#section(
  [Scientific claim],
  [
    A predictive world model can be accurate while its latent remains weakly
    informative about low-level physics. We ask whether its own reliable
    next-state prediction can supervise a low-capacity physical explanation,
    thereby inducing physical structure in the shared action-conditioned
    representation without parameter labels.
  ],
  accent: cyan,
  fill: pale-cyan,
)

#v(7pt)

#grid(
  columns: (1.02fr, 0.98fr),
  gutter: 8pt,
  [
    #section(
      [1 · Shared predictive latent],
      [
        Encode an observation history and condition it on the applied actions:

        $ z_t = E_phi(o_t), quad hat(z)_t = P_psi(z_(<=t), a_(<=t)). $

        The learned next-state head and physical probe branch from exactly the
        same $hat(z)_t$:

        $ hat(s)^A_(t+1) = D_omega(hat(z)_t), $
        $ hat(theta) = "Bound"(rho_xi("Pool"({hat(z)_t}_(t in cal(C))))). $

        $rho_xi$ is linear by default. It cannot hide another world model; it
        must express the teacher prediction through a small physical vector.
        The earlier route $rho(E(o))$ is a named pre-action ablation, not the
        proposed method.

        Infer one persistent $hat(theta)$ from causal context through index
        $K$. Fit the physical explanation on completed context transitions
        $0, dots, K-1$, then report it only on query transitions
        $K, dots, T-2$. No query outcome enters the identifier.
      ],
    )

    #v(6pt)

    #section(
      [2 · Two paths],
      [
        #grid(
          columns: (1fr, auto, 1fr),
          gutter: 4pt,
          align: center,
          [#tag[PATH A] #linebreak() learned prediction],
          [#text(size: 13pt, fill: blue)[↔]],
          [#tag(fill: pale-cyan, ink: cyan)[PATH B] #linebreak() physical explanation],
        )
        #v(4pt)
        $ hat(s)^A_(t+1) = D_omega(hat(z)_t), $
        $ hat(s)^B_(t+1) = S(s_t, a_t, hat(theta)). $

        $S$ is frozen, differentiable, and has zero learned parameters.
        Gradients pass through $S$ into the probe and, unless detached for an
        ablation, into the shared predictor representation.
      ],
      accent: cyan,
    )

    #v(6pt)

    #section(
      [3 · Teacher is the model, not a label],
      [
        Path A learns ordinary next-state prediction:

        $ cal(L)_A = norm(N(hat(s)^A_(t+1)) - N(s_(t+1)))^2. $

        Path B reproduces Path A's stopped-gradient prediction:

        $ cal(L)_B = norm(N(hat(s)^B_(t+1)) - "sg"(N(hat(s)^A_(t+1))))^2. $

        No $theta$ label appears in training, and Path B is not trained on
        $s_(t+1)$. The raw-state target is an explicit ablation.
      ],
      accent: amber,
      fill: pale-amber,
    )
  ],
  [
    #section(
      [4 · Objective and gradient contract],
      [
        $ cal(L) = cal(L)_A + alpha cal(L)_B. $

        The stop-gradient prevents $cal(L)_B$ from moving the Path-A decoder
        toward the solver. Because both heads read $hat(z)$, $cal(L)_B$ can
        still shape the predictor so that a low-capacity probe exposes a
        physical coordinate system. The post-hoc control detaches the probe
        input and therefore trains only $rho_xi$.

        #text(weight: "bold")[Checked invariants:] the solver owns no
        parameters; $theta$ is never a free optimization variable; the probe
        never sees the target frame; Path A receives no gradient through the
        stopped teacher edge.
      ],
      accent: blue,
      fill: pale-blue,
    )

    #v(6pt)

    #section(
      [5 · Paper experiment matrix],
      [
        #grid(
          columns: (auto, 1fr),
          gutter: 5pt,
          row-gutter: 3pt,
          [#tag[RQ1]], [Does Path A beat persistence while baseline physics decodability remains low?],
          [#tag[RQ2]], [Does PhysWM improve held-out physical recovery without $theta$ labels?],
          [#tag[RQ3]], [Do inferred parameters improve substitution and multi-horizon solver rollouts?],
          [#tag[ABLATE]], [pre-action latent; dataset target; detached probe input; probe capacity; context length],
          [#tag[SCALE]], [tiny CNN for diagnosis, then frozen DINOv2 for the paper result; three seeds],
        )
      ],
      accent: amber,
    )

    #v(6pt)

    #section(
      [6 · Certificates and identifiability],
      [
        #text(weight: "bold")[Prediction:] normalized held-out RMSE for Path A,
        persistence, Path B-to-teacher, and Path B-to-data.

        #text(weight: "bold")[Representation:] held-out $R^2$ from the model's
        unsupervised probe and a supervised ridge read-out on the exact same
        pooled predictive latent.

        #text(weight: "bold")[Function:] true / inferred / shuffled / nominal
        parameter substitution and multi-horizon rollout error.

        In visual-only PokeWorld, trajectories identify $k/m$, $c/m$, and the
        summed contact radius—not raw $m,k,c$ separately. Raw-parameter $R^2$
        is reported only for the visual+tactile observability condition; the
        visual-only table uses the identifiable dynamics coordinates. The
        raw-parameter condition requires a completed contact in context; its
        selector never reads a query outcome.
      ],
      accent: cyan,
      fill: pale-cyan,
    )
  ],
)
