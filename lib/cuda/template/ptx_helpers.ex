defmodule Cuda.Template.PtxHelpers do
  @moduledoc """
  Helpers available in PTX templates.
  """

  alias Cuda.{Compiler.Context, Graph.Node, Memory, Template}
  use Bitwise
  require Logger

  @doc """
  Returns true if current node compiled for back propagation graph
  """
  def back_propagation?(ctx) do
    Map.get(ctx.assigns, :back_propagation) == true ||
    Context.find_assign(ctx, [:back_propagation]) == true
  end

  @doc """
  Returns true if current node compiled for training graph
  """
  def training?(ctx) do
    Map.get(ctx.assigns, :training) == true ||
    Context.find_assign(ctx, [:training]) == true
  end

  @doc """
  Includes PTX .version directive
  """
  def version(vsn \\ "5.0")
  def version(vsn) when is_bitstring(vsn) do
    ".version #{vsn}\n"
  end
  def version({major, minor}) when is_integer(major) and is_integer(minor) do
    ".version #{major}.#{minor}\n"
  end
  def version(nil) do
    ".version 5.0\n"
  end
  def version(vsn) do
    Logger.warn("Invalid PTX version specified: `#{inspect vsn}`")
    ""
  end

  @doc """
  Includes PTX .target directive
  """
  @targets ~w(sm_60 sm_61 sm_50 sm_52 sm_53 sm_30 sm_32 sm_35 sm_37 sm_20
              sm_10 sm_11 sm_12 sm_13 texmode_unified texmode_independent
              debug map_f64_to_f32)
  def target(ctx, tgt \\ nil)
  def target(ctx, tgt) when is_list(tgt) do
    tgt
    |> Enum.map(&sanitize_target(ctx, &1))
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
    |> Enum.join(", ")
    ".target #{tgt}\n"
  end
  def target(ctx, tgt) do
    tgt = sanitize_target(ctx, tgt)
    ".target #{tgt}\n"
  end
  defp sanitize_target(_ctx, tgt) when is_bitstring(tgt) and tgt in @targets do
    tgt
  end
  defp sanitize_target(_ctx, nil) do
    # TODO: when we get target from GPU (sm_35 for example) following
    #       compilation error occured:
    #
    #         SM version specified by .target is higher than default SM
    #         version assumed
    #
    # We need to find way to compile with detected targets
    {major, minor} = {2, 0} #ctx.env.gpu_info[:compute_capability]
    "sm_#{major}#{minor}"
  end
  defp sanitize_target(ctx, tgt) when is_atom(tgt) do
    sanitize_target(ctx, "#{tgt}")
  end
  defp sanitize_target(ctx, {major, minor}) when is_integer(major) and is_integer(minor) do
    sanitize_target(ctx, "sm_#{major}#{minor}")
  end
  defp sanitize_target(ctx, tgt) when not is_nil(tgt) do
    default = sanitize_target(ctx, nil)
    Logger.warn("Invalid PTX target specified: `#{inspect tgt}`")
    Logger.warn("Default PTX target `#{default}` will be used")
    default
  end

  @doc """
  Includes PTX .address_size directive
  """
  def address_size(ctx, size \\ nil)
  def address_size(_ctx, size) when size in [32, 64] do
    ".address_size #{size}\n"
  end
  def address_size(ctx, nil) do
    size = ctx.env.gpu_info[:global_memory_bus_width]
    ".address_size #{size}\n"
  end
  def address_size(ctx, size) do
    default = ctx.env.gpu_info[:global_memory_bus_width]
    Logger.warn("Invalid PTX address size specified: `#{inspect size}`")
    Logger.warn("Default PTX address size `#{default}` will be used")
    ".address_size #{default}\n"
  end

  @doc """
  Includes PTX directives header that includes .version, .target and
  .address_size directives
  """
  def header(ctx, vsn \\ nil, tgt \\ nil, size \\ nil) do
    version(vsn) <> target(ctx, tgt) <> address_size(ctx, size)
  end

  @doc """
  Returns offset of variable in specified memory block
  """
  def offset(ctx, memory, var) do
    #IO.inspect({memory, var})
    shape = Context.find_assign(ctx, [:memory, memory], ctx.path, &has_var?(&1, var))
    shape = with nil <- shape do
      get_in(ctx.assigns, [:memory, memory])
    end# |> IO.inspect
    with nil <- Memory.offset(shape, var) do
      Logger.warn("Can't find offset for `#{inspect var}` in memory `#{memory}`")
      nil
    end
  end

  @doc """
  Returns offset of variable in shared memory block
  """
  def shared_offset(ctx, var) do
    #IO.inspect(ctx)
    case Template.Helpers.var(ctx, :layer) do
      nil   -> raise CompileError, description: "Layer variable is not defined"
      layer -> offset(ctx, :shared, [var, layer])
    end
  end

  @doc """
  Returns offset of specified pin
  """
  def pin_offset(ctx, var) do
    offset(ctx, :pins, var)
  end

  @doc """
  Defines PTX kernel function
  """
  def kernel(ctx, name, body, opts \\ []) do
    params = [{:pins, :u64, [ptr: true]}] ++ Keyword.get(opts, :args, [])
    params = params |> Enum.map(&param/1) |> Enum.join(", ")
    ".visible .entry #{current_node_id(ctx)}__#{name} (#{params}) {\n" <>
    body <>
    "\n}"
  end

  @doc """
  Includes specified include-module
  """
  def include(ctx, module, part \\ :body, opts \\ []) do
    {part, opts} = case part do
      opts when is_list(opts) -> {:body, opts}
      part                    -> {part, opts}
    end
    with {:module, _} <- Code.ensure_loaded(module) do
      case function_exported?(module, :__ptx__, 2) do
        true -> module.__ptx__(part, Keyword.put(opts, :ctx, ctx))
        _    -> ""
      end
    else
      _ -> raise CompileError, description: "Couldn't compile include module #{module}"
    end
  end

  def param({name, type, opts}) do
    space = opts
            |> Keyword.take(~w(const global local shared)a)
            |> Enum.reduce([], fn
              {name, true}, [] -> [".#{name}"]
              _, acc           -> acc
            end)
    align = opts
            |> Keyword.take(~w(align)a)
            |> Enum.reduce([], fn
              {:align, x}, _ when band(x, x - 1) == 0 -> [".align #{x}"]
              _, acc                                  -> acc
            end)
    param = [".param", ".#{type}"] ++
            (if Keyword.get(opts, :ptr) == true, do: [".ptr"], else: []) ++
            space ++
            align ++
            ["#{name}"]
    param |> Enum.join(" ")
  end

  defmacro back_propagation?() do
    quote do
      back_propagation?(var!(ctx))
    end
  end

  defmacro training?() do
    quote do
      training?(var!(ctx))
    end
  end

  defmacro offset(memory, var) do
    quote do
      offset(var!(ctx), unquote(memory), unquote(var))
    end
  end

  defmacro shared_offset(var) do
    quote do
      shared_offset(var!(ctx), unquote(var))
    end
  end

  defmacro pin_offset(var) do
    quote do
      pin_offset(var!(ctx), unquote(var))
    end
  end

  defmacro defkernel(ctx, name, args, opts) do
    body = Keyword.get(opts, :do)
    args = args
           |> Enum.map(&parse_arg/1)
           |> Enum.filter(&is_tuple/1)
           |> Macro.escape
    quote do
      kernel(unquote(ctx), unquote(name), unquote(body), args: unquote(args))
    end
  end
  defmacro defkernel(ctx, name, opts) do
    body = Keyword.get(opts, :do)
    quote do
      kernel(unquote(ctx), unquote(name), unquote(body))
    end
  end

  defp has_var?(%Memory{} = memory, [key | path]) do
    with %{} = map <- Memory.get(memory, key) do
      get_in(map, path)
    else
      _ -> false
    end
  end
  defp has_var?(%Memory{} = memory, var) do
    Memory.has_key?(memory, var)
  end
  defp has_var?(map, path) when is_list(path) do
    get_in(map, path)
  end
  defp has_var?(map, var) do
    Map.has_key?(map, var)
  end

  defp current_node_id(ctx) do
    Node.string_id(Map.get(Context.node(ctx) || %{}, :id))
  end

  defp parse_arg(arg, opts \\ [])
  defp parse_arg({name, type}, opts) when is_atom(type) do
    {name, type, opts}
  end
  defp parse_arg({name, {{:., _, [{type, _, x}, opt]}, _, _}}, opts) when is_atom(x) do
    {name, type, [{opt, true} | opts]}
  end
  defp parse_arg({name, {{:., _, [nested, opt]}, _, _}}, opts) do
    parse_arg({name, nested}, [{opt, true} | opts])
  end
  defp parse_arg({name, {:-, _, [{{:., _, [nested, opt]}, _, _}, v]}}, opts) do
    parse_arg({name, nested}, [{opt, v} | opts])
  end
  defp parse_arg({name, {type, _, _}}, opts) when is_atom(type) do
    {name, type, opts}
  end
  defp parse_arg(_, _) do
    nil
  end
end
