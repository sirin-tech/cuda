defmodule Cuda.Template do
  @moduledoc """
  Represents eex templates processing module
  """

  @type options :: [
    context: Cuda.Template.Context.t,
    ptx_helpers: [],
    c_helpers: []]

  @doc """
  Preprocesses PTX template
  """
  @spec ptx_preprocess(template :: String.t, opts :: options) :: String.t
  def ptx_preprocess(template, opts) do
    ptx_eval(template, opts)
  end

  @doc """
  Preprocesses C template
  """
  @spec c_preprocess(template :: String.t, opts :: options) :: String.t
  def c_preprocess(template, opts) do
    c_eval(template, opts)
  end

  @doc """
  Returns evaluated PTX template with included helper modules, and etc.
  """
  @spec ptx_eval(template :: String.t, opts :: options) :: String.t
  def ptx_eval(template, opts) do
    hlprs = [Cuda.Template.Helpers | Keyword.get(opts, :ptx_helpers)]
    ctx = Keyword.get(opts, :context)
    eval(template, ctx, hlprs)
  end

  @doc """
  Returns evaluated C template with included helper modules, and etc.
  """
  @spec c_eval(template :: String.t, opts :: options) :: String.t
  def c_eval(template, opts) do
    hlprs = [Cuda.Template.Helpers | Keyword.get(opts, :c_helpers)]
    ctx = Keyword.get(opts, :context)
    eval(template, ctx, hlprs)
  end

  defp eval(template, context, helpers) do
    func = get_funcs(helpers)
    EEx.eval_string(template, [ctx: context], [functions: func])
  end

  defp get_funcs([]), do: []
  defp get_funcs([module | rest]) do
    {:ok, funcs} = get_funcs(module)
    [funcs | get_funcs(rest)]
  end
  defp get_funcs(module) do
    funcs = module.__info__(:functions)
    {:ok, {module, funcs}}
  end
end
