:layout: landing
:description: "swydd will yield desired deliverables"

swydd
=====

.. rst-class:: lead
    
    Swydd will yield desired deliverables.
    See the `api reference </api.html>`_ for more info.
    There is not currently sufficient usage examples.
    For now check `tasks.py <https://github.com/daylinmorgan/swydd/blob/main/tasks.py>`_ for example usage


Automagic Snippet
-----------------

.. code-block:: python

    if not (
      (_i := __import__)("importlib.util").util.find_spec("swydd")
      or (_src := _i("pathlib").Path(__file__).parent / "swydd/__init__.py").is_file()
    ): # noqa | https://github.com/daylinmorgan/swydd?tab=readme-ov-file#automagic-snippet
      _r = _i("urllib.request").request.urlopen("https://swydd.dayl.in/swydd.py")
      _src.parent.mkdir(exist_ok=True)
      _src.write_text(_r.read().decode())

.. toctree::
   :hidden:

   api
