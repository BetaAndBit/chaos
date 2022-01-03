---
title: "Fraktale - język Julia"
output:
  html_document:
    keep_md: yes
    toc: true
    toc_depth: 2
    toc_float: true
    df_print: paged
---

# Dywan Sierpińskiego

## Narysuj mi kwadrat

```julia
function rect(x1,y1, x2,y2)
    return [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
end
```

```julia
using Plots

Plots.plot(rect(0,0,1,1),seriestype=:shape,legend=:false,color=:black,
           xlims=(0-0.05,1+0.05),ylims=(0-0.05,1+0.05))
```
<figure>
<center>
<img alt="png" src="kwadrat1_jl.png">
</center>
</figure>

```julia
using Plots

Plots.plot(rect(0, 0, 1/3, 1/3), seriestype=:shape,legend=:false,color=:black,
           xlims=(0-0.05,1+0.05),ylims=(0-0.05,1+0.05))
Plots.plot!(rect(0+2/3, 0, 1/3+2/3, 1/3), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0, 0+2/3, 1/3, 1/3+2/3), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/3, 0+2/3, 1/3+2/3, 1/3+2/3), seriestype=:shape,legend=:false,color=:black)
```
<figure>
<center>
<img alt="png" src="kwadrat4_jl.png">
</center>
</figure>

```julia
using Plots

Plots.plot(rect(0, 0, 1/9, 1/9), seriestype=:shape,legend=:false,color=:black,
           xlims=(0-0.05,1+0.05),ylims=(0-0.05,1+0.05))
Plots.plot!(rect(0+2/9, 0, 1/9+2/9, 1/9), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0, 0+2/9, 1/9, 1/9+2/9), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/9, 0+2/9, 1/9+2/9, 1/9+2/9), seriestype=:shape,legend=:false,color=:black)

Plots.plot!(rect(0+2/3, 0, 1/9+2/3, 1/9), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/3+2/9, 0, 1/9+2/3+2/9, 1/9), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/3, 0+2/9, 1/9+2/3, 1/9+2/9), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/3+2/9, 0+2/9, 1/9+2/3+2/9, 1/9+2/9), seriestype=:shape,legend=:false,color=:black)

Plots.plot!(rect(0, 0+2/3, 1/9, 1/9+2/3), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/9, 0+2/3, 1/9+2/9, 1/9+2/3), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0, 0+2/3+2/9, 1/9, 1/9+2/3+2/9), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/9, 0+2/3+2/9, 1/9+2/9, 1/9+2/3+2/9), seriestype=:shape,legend=:false,color=:black)

Plots.plot!(rect(0+2/3, 0+2/3, 1/9+2/3, 1/9+2/3), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/3+2/9, 0+2/3, 1/9+2/3+2/9, 1/9+2/3), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/3, 0+2/3+2/9, 1/9+2/3, 1/9+2/3+2/9), seriestype=:shape,legend=:false,color=:black)
Plots.plot!(rect(0+2/3+2/9, 0+2/3+2/9, 1/9+2/3+2/9, 1/9+2/3+2/9), seriestype=:shape,legend=:false,color=:black)
```
<figure>
<center>
<img alt="png" src="kwadrat16_jl.png">
</center>
</figure>

## Rekurencja

```julia

```

## Prawdziwy dywan

```julia

```

## Bardzo bardzo mały kwadrat wygląda jak punkt

```julia

```

# Trójkąt Sierpińskiego

## Zróbmy to w R!

```julia
function trojkat(x, y, bok)
    return [(x+bok*0,y+bok*0),(x+bok*1,y+bok*0),(x+bok*1/2,y+bok*sqrt(2)/2),(x+bok*0,y+bok*0)]
end

```

## Bardzo bardzo mały trójkąt wygląda jak punkt

```julia

```


# Gra w chaos

## Wylosuj mi transformacje

```julia
using StatsBase
using Plots

function function1(x,y)
   return x/2, y/2
end

function function2(x,y)
  return x/2 + 1/2, y/2
end

function function3(x,y)
  return x/2 + 1/4, y/2 + sqrt(3)/4
end

functions = [function1,function2,function3]

N = 30000
x, y = 0, 0
x_value = []
y_value = []

for i in 1:N
  fun = sample(functions,1)
  x, y = fun[1](x,y)
  x_value = push!(x_value,x)
  y_value = push!(y_value,y)
end

Plots.scatter(x_value, y_value, legend=:false, color=:black,
              markerstrokecolor=:white, markeralpha=0.4, markersize=2)
```
<figure>
<center>
<img alt="png" src="transf_jl.png">
</center>
</figure>

## Paproć Barnsleya

```Julia
using StatsBase
using Plots

function function1(x,y)
   return 0, 0.16*y
end

function function2(x,y)
  return 0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6
end

function function3(x,y)
  return 0.2*x - 0.26*y, 0.23*x + 0.22*y + 1.6
end

function function4(x,y)
  return -0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44
end

functions = [function1, function2, function3, function4]

N = 50000
x, y = 0, 0
x_value = []
y_value = []

for i in 1:N
  fun = sample(functions,Weights([0.01, 0.85, 0.07, 0.07]))
  x, y = fun(x,y)
  x_value = push!(x_value,x)
  y_value = push!(y_value,y)
end

Plots.scatter(x_value, y_value, legend=:false, color=:black,
              markerstrokecolor=:white, markeralpha=0.4, markersize=2)
```
<figure>
<center>
<img alt="png" src="paproc_jl.png">
</center>
</figure>

## Algebra, wszędzie algebra

```Julia
using Plots

function fu(x,p)
   if p <= 0.01
     m = [0 0;  0 0.16]
     f = [0 0]'
     return m * x + f
   elseif p <= 0.86
     m = [0.85 0.04;  -0.04 0.85]
     f = [0 1.6]'
     return m * x + f
   elseif p <= 0.93
     m = [0.20 -0.26;  0.23 0.22]
     f = [0 1.6]'
     return m * x + f
   elseif p > 0.93
     m = [-0.15 0.28;  0.26 0.24]
     f = [0 0.44]'
     return m * x + f
   end
end

N = 50000
p = rand(N);
x, y = 0, 0
x_value = []
y_value = []

for i in 1:N
  x, y = fu([x,y],p[i])
  x_value = push!(x_value,x)
  y_value = push!(y_value,y)
end

Plots.scatter(x_value, y_value, legend=:false, color=:black,
              markerstrokecolor=:white, markeralpha=0.4, markersize=2)
```
<figure>
<center>
<img alt="png" src="algebra_jl.png">
</center>
</figure>

## Smok Heighwaya

```Julia
using Plots

function gu(x,p)
   if p <= 0.5
     m = [-0.4 0;  0 -0.4]
     f = [-1 0.1]'
     return m * x + f
   else
     m = [0.76 -0.4;  0.4 0.76]
     f = [0 0]'
     return m * x + f
   end
end

N = 100000
p = rand(N);
x, y = 0, 0
x_value = []
y_value = []

for i in 1:N
  x, y = gu([x,y],p[i])
  x_value = push!(x_value,x)
  y_value = push!(y_value,y)
end

Plots.scatter(x_value, y_value, legend=:false, color=:black,
              markerstrokecolor=:white, markeralpha=0.4, markersize=2)
```
<figure>
<center>
<img alt="png" src="smok_jl.png">
</center>
</figure>
