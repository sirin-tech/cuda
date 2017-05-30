defmodule Cuda.Template do
  @moduledoc """
  Represents eex templates processing module
  """

  @type options :: [
    context: Cuda.Compiler.Context.t,
    ptx_helpers: [], # ? - subject to remove - use helpers option
    c_helpers: [],   # ? - subject to remove - use helpers option
    helpers: []]

  @doc """
  Preprocesses PTX template
  """
  @spec ptx_preprocess(template :: String.t, opts :: options) :: {:ptx, String.t}
  def ptx_preprocess(template, opts) do
    {:ptx, ptx_eval(template, opts)}
  end

  @doc """
  Preprocesses C template
  """
  @spec c_preprocess(template :: String.t, opts :: options) :: {:c, String.t}
  def c_preprocess(template, opts) do
    {:c, c_eval(template, opts)}
  end

  @doc """
  Returns evaluated PTX template with included helper modules, and etc.
  """
  @spec ptx_eval(template :: String.t, opts :: options) :: String.t
  def ptx_eval(template, opts) do
    hlprs = [Kernel, Cuda.Template.Helpers] ++
             Keyword.get(opts, :ptx_helpers, []) ++
             Keyword.get(opts, :helpers, [])
    ctx = Keyword.get(opts, :context)
    eval(template, ctx, hlprs)
  end

  @doc """
  Returns evaluated C template with included helper modules, and etc.
  """
  @spec c_eval(template :: String.t, opts :: options) :: String.t
  def c_eval(template, opts) do
    hlprs = [Kernel, Cuda.Template.Helpers] ++
             Keyword.get(opts, :c_helpers, []) ++
             Keyword.get(opts, :helpers, [])
    ctx = Keyword.get(opts, :context)
    eval(template, ctx, hlprs)
  end

  defp eval(template, context, helpers) do
    opts = [functions: get_funcs(helpers),
            macros: get_macros(helpers)]
    EEx.eval_string(template, [ctx: context, assigns: context.assigns], opts)
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

  defp get_macros([]), do: []
  defp get_macros([module | rest]) do
    {:ok, macros} = get_macros(module)
    [macros | get_macros(rest)]
  end
  defp get_macros(module) do
    macros = module.__info__(:macros)
    #macros = case module do
    #  Kernel -> macros |> Enum.reject(fn {n, _} -> n == :@ end)
    #  _      -> macros
    #end
    {:ok, {module, macros}}
  end
end
