Nim
==========================

This script implements two evalution startegies, namely $(1+\lambda)$ and $(1,\lambda)$, to find the best strategy to win at [Nim](https://en.wikipedia.org/wiki/Nim).

The game supports two versions: *normal game* and *misère play*. In *normal game* the player who takes the last object wins. Instead, if Nim is played as a *misère game*, the player to take the last object loses.

Both versions of the game, along with the optimal strategies to win, are implemented.

For what concernes the evolution strategies, $(1+\lambda)$ is implemented both in the classical version and with self adaptive parameter $\sigma$, while $(1,\lambda)$ is implemented in the classical version.

Both $(1+\lambda)$ and $(1,\lambda)$ show convergence to the optimal strategy, using as parameter $\lambda=20, \sigma=0.05,$ *max_iterations*=5000$ and  $\lambda=50, \sigma=0.05$, *max\_iterations*=5000, respectively.
