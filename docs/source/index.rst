:layout: landing
:description: "swydd will yield desired deliverables"

swydd
=====

.. rst-class:: lead

    **S**\ wydd **w**\ ill **y**\ ield **d**\ esired **d**\ eliverables.
    See the `api reference </api.html>`_ for more info.
    There is not currently sufficient usage examples.
    For now check `tasks.py <https://github.com/daylinmorgan/swydd/blob/main/tasks.py>`_ for example usage


Automagic Snippet
-----------------

.. code-block:: python

  # fmt: off
  # https://swydd.dayl.in/#automagic-snippet
  if not((_i:=__import__)("importlib.util").util.find_spec("swydd")or
  (_src:=_i("pathlib").Path(__file__).parent/"swydd/__init__.py").is_file()):
    _r=_i("urllib.request").request.urlopen("https://swydd.dayl.in/swydd.py")
    _src.parent.mkdir(exist_ok=True);_src.write_text(_r.read().decode())  # noqa
  # fmt: on

.. toctree::
   :hidden:

   api
