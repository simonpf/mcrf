# A framework for microwave cloud retrievals

The Microwave Cloud Retrieval Framework (MCRF) is a framework written in Python
for performing cloud retrieval using observations from radars and passive
radiometers. The main purpose of this project is to serve as a proof-of-concept
for the retrieval of frozen hydrometeors from sub-millimeter wave observations
combined with radar observations.

## Dependencies

This package builds on the [ARTS](http://radiativetransfer.org) package for
performing the simulations of radiative transfer as well as the retrievals. In
addition to that, the following Python packages are required:
- [typhon](https://github.com/atmtools/typhon)
- [parts](https://github.com/simonpf/parts)
