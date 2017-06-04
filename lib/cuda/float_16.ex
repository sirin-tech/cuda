defmodule Cuda.Float16 do
  @moduledoc """
  Represents Float 16 value
  Source: http://www.softelectro.ru/ieee754.html
  """

  # TODO: При проведении тестов с параметрами Float_32, выяснилось что для дробных
  #       значений младшие биты мантиссы отличаются от системных значений, например:
  #        - Мантисса числа 0.1234567 после системного преобразования:     11111001101011011011110
  #        - Мантисса числа 0.1234567 после преобразования данным модулем: 11111001101011011011101

  #======Float_16======
  # sign     - 1  bit
  # exponent - 5  bits
  # mantiss  - 10 bits
  # Exp bias - 15
  #
  #======Float_32======
  # sign     - 1  bit
  # exponent - 8  bits
  # mantiss  - 23 bits
  # Exp bias - 127
  #
  #====================
  use Bitwise

  @sign_size     1
  @exponent_size 5
  @mantiss_size  10
  @exponent_bias 15

  @pow_2_minus_14 0.000061035
  @pow_2_10 1024
  @min_normal 0.000061

  @doc """
  Converts number to binary representation of float 16 type
  """
  @spec pack(number) :: binary
  def pack(x) when is_integer(x), do: pack(x + 0.0)
  # subnormal converting
  def pack(x) when abs(x) < @min_normal do
    # Sign bit encode
    {x, sign} = sign_encode(x)
    # gets binary fractional number
    {_, {m, s}} = float_binary(x / @pow_2_minus_14)
    # cut binary to fit mantiss size
    m = m >>> (s - @mantiss_size)
    # pack converted value to binary
    <<sign::size(@sign_size), 0::size(@exponent_size), m::size(@mantiss_size)>>
  end
  # normal converting
  def pack(x) do
    # Sign bit encode
    {x, sign} = sign_encode(x)
    # gets binary fractional number
    f = float_binary(x)
    # gets normalized mantiss and exponent
    {m, e} = normalize(f, @exponent_bias)
    n = size(m) - 1
    # у мантиссы убирается ведущая 1 (старший бит)
    m_cutted = m - (1 <<< n)
    m_cutted = if (@mantiss_size - n) >= 0 do
      # мантисса дополняется младшими разрядами до необходимой разрядности типа
      m_cutted <<< (@mantiss_size - n)
    else
      # если разрядность мантиссы больше установленной то, ее младшие биты
      # обрезаются  до необходимых размеров
      m_cutted >>> (n - @mantiss_size)
    end
    # pack converted value to binary
    <<sign::size(@sign_size), e::size(@exponent_size), m_cutted::size(@mantiss_size)>>
  end

  @doc """
  Converts binary representation of float 16 type to float value
  """
  @spec unpack(binary) :: float
  # if exponent == 11111 and mantiss == 0, then it's represents positive or
  # negative infinity
  def unpack(<<sign::size(@sign_size), 31::size(@exponent_size), 0::size(@mantiss_size)>>), do: sign == 0 && :positive_infinity || :negative_infinity
  # if exponent == 11111 and mantiss != 0, then it's represents not a number value
  def unpack(<<_::size(@sign_size),    31::size(@exponent_size), _::size(@mantiss_size)>>), do: :not_a_number
  # if exponent == 0 and mantiss = 0, then it's represents 0.0
  def unpack(<<_::size(@sign_size),    0::size(@exponent_size),  0::size(@mantiss_size)>>), do: 0.0
  # subnormal values
  def unpack(<<sign::size(@sign_size), 0::size(@exponent_size), m::size(@mantiss_size)>>) do
    # Sign bit decode
    sign = sign_decode(sign)
    # Formula explains at http://www.softelectro.ru/ieee754.html paragraph 4.2
    #
    # При изменении параметров типа в формуле необходимо изменить @pow_2_10
    # на 1 <<< @mantiss_size
    sign * @pow_2_minus_14 * m/@pow_2_10
  end
  # normalized values
  def unpack(<<sign::size(@sign_size), exp::size(@exponent_size), m::size(@mantiss_size)>>) do
    # Sign bit decode
    sign = sign_decode(sign)
    # Formula explains at http://www.softelectro.ru/ieee754.html paragraph 4.2
    #
    # При изменении параметров типа в формуле необходимо изменить @pow_2_10
    # на 1 <<< @mantiss_size
    sign * :math.pow(2, exp - @exponent_bias) * (1 + m/@pow_2_10)
  end

  # Encodes sign of the value, and turn value absolute for further convertations
  defp sign_encode(x) when x < 0, do: {abs(x), 1}
  defp sign_encode(x), do: {x, 0}

  # Decodes sign bit
  defp sign_decode(0), do: 1
  defp sign_decode(1), do: -1

  # Converts float value to the binary representation
  # @type float_binary :: {whole_part :: integer, fractional_part :: binary}
  # @spec float_binary(value :: float, precision :: integer) :: float_binary
  defp float_binary(value, precision \\ 50) when is_float(value) do
    # get the whole part of value
    i = round(Float.floor(value))
    # get the fractional part of value
    fract = value - i
    f = fb_fract_binary(fract, precision)
    {i, f}
  end

  # Normalizes binary fractional number to mantiss and exponent
  # @type normalize :: {mantiss :: binary, exponent :: integer}
  # @spec normalize(value :: float_binary, bias :: integer) :: normalize
  defp normalize(value, bias)
  defp normalize({0, {0, _}}, _), do: {0, 0}
  defp normalize({0, {fract, fract_size}}, bias) do
    n = size(fract)
    # get exponent with bias
    exp = -(fract_size - n + 1) + bias
    {fract, exp}
  end
  defp normalize({int, {fract, fract_size}}, bias) do
    int_n = size(int)
    # get exponent with bias
    exp = int_n - 1 + bias
    # целое сдвигается влево на кол-во разрядов дробной части,
    # добавляется дробная часть
    int = (int <<< fract_size) ||| fract
    {int, exp}
  end

  # Gets binary digits count
  defp size(0), do: 0
  defp size(x) do
    round(Float.floor(:math.log2(x))) + 1
  end

  # Implements algorithm from this source https://otvet.mail.ru/question/46720675
  # Перевод из десятичной системы счисления в двоичную и шестнадцатеричную:
  # а) исходная дробь умножается на основание системы счисления, в которую
  #    переводится (2 или 16);
  # б) в полученном произведении целая часть преобразуется в соответствии с
  #    таблицей в цифру нужной системы счисления и отбрасывается – она является
  #    старшей цифрой получаемой дроби;
  # в) оставшаяся дробная часть (это правильная дробь) вновь умножается на
  #    нужное основание системы счисления с последующей обработкой полученного
  #    произведения в соответствии с шагами а) и б);
  # г) процедура умножения продолжается до тех пор, пока ни будет получен
  #    нулевой результат в дробной части произведения или ни будет достигнуто
  #    требуемое количество цифр в результате;
  # д) формируется искомое число: последовательно отброшенные в шаге б) цифры
  #    составляют дробную часть результата, причем в порядке уменьшения старшинства.
  #
  # fb_fract_binary returns:
  # {fractional_result, overall_digits_count}
  defp fb_fract_binary(fract, precision, result \\ {0, 0})
  defp fb_fract_binary(0.0, _, result), do: result
  defp fb_fract_binary(_,   0, result), do: result
  defp fb_fract_binary(fract, precision, {result, digits}) do
    fract = fract * 2
    result = result <<< 1
    if fract >= 1 do
      fb_fract_binary(fract - 1, precision - 1, {result + 1, digits + 1})
    else
      fb_fract_binary(fract, precision - 1, {result, digits + 1})
    end
  end

  def dbg_view(data, binary \\ true)
  def dbg_view(<<sign::size(@sign_size), exp::size(@exponent_size), m::size(@mantiss_size)>>, binary) do
    base = binary && :binary || :decimal
    IO.inspect(sign, label: :sign, base: base)
    IO.inspect(exp, label: :exponent, base: base)
    IO.inspect(m, label: :mantiss, base: base)
    :ok
  end
  def dbg_view(_, _), do: raise(ArgumentError, message: "Value has wrong format")
end
