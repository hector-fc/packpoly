
using JuMP
import Ipopt
import Random
using Printf
using CairoMakie 
using GeometryBasics

function pkFun(n,x₀,y₀,a₀,ns₀,b₀,rr,V,rd)    
  pkPoly2D = Model(Ipopt.Optimizer)  
  @variable(pkPoly2D,-10<= x[i=1:n]<=10, start = x₀[i])
  @variable(pkPoly2D,-10<=y[i=1:n]<=10, start = y₀[i])   
  @variable(pkPoly2D, 0<=a[i=1:n]<=2π, start = a₀[i])
  @variable(pkPoly2D,0<=ns[i=1:n,j=1:n]<=2π, start = ns₀[i,j])
  @variable(pkPoly2D,-100<=b[i=1:n,j=1:n]<=100,start = b₀[i,j])
  @NLobjective(pkPoly2D,Min,0.0)
  @NLconstraint(pkPoly2D, 
    l[i=1:n,j=1:n, k=1:V; i<j], 
      cos(ns[i,j])*(x[i]+rr*cos(k*2π/V + a[i])) + 
      sin(ns[i,j])*(y[i]+rr*sin(k*2π/V + a[i])) - b[i,j] <=0 
    ) 
  @NLconstraint(pkPoly2D, 
    r[i=1:n,j=1:n, k=1:V; i<j], 
    b[i,j]- (cos(ns[i,j])*(x[j]+rr*cos(k*2π/V + a[j])) 
    + sin(ns[i,j])*(y[j]+rr*sin(k*2π/V+a[j])))<=0 
    ) 

  @NLconstraint(pkPoly2D, l1[i=1:n, k=1:V], 
    (x[i]+rr*cos(k*2π/V + a[i]))^2 + 
    (y[i]+rr*sin(k*2π/V + a[i]))^2 <= rd^2)

  set_silent(pkPoly2D)
  solTime=@elapsed optimize!(pkPoly2D);  
  return pkPoly2D,x,y,a,ns,b,solTime  
end 

# ======= 

function drawJ(n, x, y, a, rr, V, nPrint, rd)   
  fig = Figure(resolution = (600, 600))
  ax = Axis(fig[1, 1]; aspect = DataAspect())        
  polygons = Vector{Vector{Point2f}}()
  colors = RGBAf[]
  for i in 1:n      
      c = RGBAf(rand(), rand(), rand(), 0.3)
      push!(colors, c)      
      pts = Point2f[]
      for ii in 1:V
          θ = ii * 2π / V + value.(a[i])
          xf = value.(x[i]) + rr * cos(θ)
          yf = value.(y[i]) + rr * sin(θ)            
          push!(pts, Point2f(xf, yf))
      end
      push!(polygons, pts)        
  end
  poly!(ax, polygons; color = colors, strokecolor = :black, strokewidth = 0.5)

  θs = range(0, 2π, length = 200)
  xs = rd * cos.(θs)
  ys = rd * sin.(θs)
  lines!(ax, xs, ys; color = :black, linewidth = 1)
  limits!(ax, -rd - rr, rd + rr, -rd - rr, rd + rr)
  namefile = "drawJ"*nPrint*".png"
  save(namefile, fig)
  #return fig
end
# ======= 
function den(n,V,r,L) 
  # density 
  num = n*V*r^2*sin(2.0*π/V)/2.0 
  den = π*L^2 
  return num/den   
end

function upB(V,r,L) 
  # upper bound 
  num = π*L^2 
  den = V*r^2*sin(2.0*π/V)/2.0 
  return num/den 
end

function dataS(file,iter,n,V,r,L,TrialN,solTime)
  temp0 = den(n,V,r,L) 
  temp1 = upB(V,r,L) 
  io = open(file,"w")  
  @printf(io,"%d & %d  & %d  & %4.2f & %8.4f & %4.2f & %d & %12.4f \n",
              iter, n,    V,    r,  temp0,temp1,TrialN,solTime) 
  close(io)
end 

# ======= 

function  packOne(numP,rr,Vert,rd)    
  file =  "dataSol.txt"
  io = open(file,"w")  
  close(io)

  timeLimit= 60.0  #  50 seconds   10 minutos
  startTime= time()

  solTime  = 0.0   
  TrialP = 0   
  TrialN = 0   
  #iterMax = 10 
  V = Vert #
  n  = numP 
  temp = true
  iter = 1          
  ϵ = 0.3 
  xt, yt, at, nst, bt = nothing, nothing, nothing, nothing, nothing

  while  ( time()- startTime) < timeLimit 
    if n == 2 
      x₀ = -0.5*ϵ .+ ϵ*rand(n)
      y₀ = -0.5*ϵ .+ ϵ*rand(n)
      a₀ =  2π*rand(n)
      ns₀ = 2π*rand(n,n) 
      b₀ = -0.5*ϵ .+ ϵ*rand(n,n)   

      model,xt,yt,at,nst,bt,solT = pkFun(n,x₀,y₀,a₀,ns₀,b₀,rr,V,rd) 
      solTime = solTime + solT 

      temp = termination_status(model) == LOCALLY_SOLVED
      if temp        
        drawJ(n,xt,yt,at,rr,V,"sol",rd)
        n = n + 1 
      end          
      iter = iter + 1            
    elseif temp        
      x₀ =ϵ*rand(n)   #ϵ*
      y₀ =ϵ*rand(n)
      a₀ =ϵ*rand(n)
      ns₀=ϵ*rand(n,n) 
      b₀ =ϵ*rand(n,n)   

      x₀[1:n-1] = value.(xt)
      y₀[1:n-1] = value.(yt)
      a₀[1:n-1] = value.(at)
      ns₀[1:n-1,1:n-1] = value.(nst)
      b₀[1:n-1,1:n-1] = value.(bt) 
      
      model,xt,yt,at,nst,bt,solT = pkFun(n,x₀,y₀,a₀,ns₀,b₀,rr,V,rd)       
      solTime = solTime + solT 
      temp = termination_status(model) == LOCALLY_SOLVED
      if temp        
        drawJ(n,xt,yt,at,rr,V,"sol",rd)
        dataS(file,iter,n,V,rr,rd,TrialN,solTime)        
        TrialP = TrialP + 1 
        n = n + 1 
      end          

    else                 
      x₀ =  ϵ*value.(xt)
      y₀ =  ϵ*value.(yt)
      a₀ =  ϵ*value.(at)
      ns₀=  ϵ*value.(nst)
      b₀ =  ϵ*value.(bt) 

      model,xt,yt,at,nst,bt,solT = pkFun(n,x₀,y₀,a₀,ns₀,b₀,rr,V,rd)       
      solTime = solTime + solT 
      TrialN = TrialN + 1 
      temp = termination_status(model) == LOCALLY_SOLVED
      if !temp 
        TrialN = TrialN +1 
      end 
      if temp        
        drawJ(n,xt,yt,at,rr,V,"sol",rd)
        dataS(file,iter,n,V,rr,rd,TrialN,solTime)        
        n = n + 1 
      end                
    end
    iter = iter + 1      
    @printf("iIter: %d, Itens: %d TrialN: %d time: %4.2f \n",
    iter,n,TrialN,solTime)  
  end  
  @printf("Were Packed %d time: %2.1f\n",n-1,solTime)
end 

packOne(2,1.0,5,4)    

