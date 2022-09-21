using Plots

function trojkat(x, y, bok)
    return [(x+bok*0,y+bok*0),(x+bok*1,y+bok*0),(x+bok*1/2,y+bok*sqrt(2)/2),(x+bok*0,y+bok*0)]
end

# czyścimy ekran
E = 0.05
Plots.plot(0,xlim=(0-E,1+E),ylim=(0-E,1+E-0.1))

function rysuj_uszczelke(x, y, szerokosc, iteracja)
  if iteracja == 0
      Plots.plot!(trojkat(x, y, szerokosc), seriestype=:shape, color=:black, legend=:false)
  else
      rysuj_uszczelke(x, y, szerokosc / 2, iteracja - 1)
      rysuj_uszczelke(x+szerokosc / 2, y, szerokosc / 2, iteracja - 1)
      rysuj_uszczelke(x+szerokosc / 4, y+sqrt(3)*szerokosc/4, szerokosc / 2, iteracja - 1)
  end
current()
end

rysuj_uszczelke(0, 0, 1, 5)