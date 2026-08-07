"""Groundwire desktop app — Briefcase launcher package.

`groundwire/__main__.py` is the CLI harness, so the packaged app can't just run
`python -m groundwire`. Briefcase launches `python -m gwapp` instead, and this
package hands off to the native desktop window.
"""
