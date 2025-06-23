
using JuMP
import Ipopt
import Random
import Statistics
using Printf
using CairoMakie
using GeometryBasics


#=
vt = [ 1  1 -1 -1;
       1 -1  1 -1;
       1 -1 -1  1] 
=#


function pkFun(nP,x₀,y₀,ax₀,ay₀,b₀,R₀,rr,V,rd)    
cos_row = [cos(k * 2π / V) for k in 0:(V-1)]
sin_row = [sin(k * 2π / V) for k in 0:(V-1)] 
vt = hcat(cos_row, sin_row)
vt= rr*transpose(vt)
nV = size(vt[1,:])[1]

# Create a model
model = Model(Ipopt.Optimizer)

@variable(model, R[t=1:nP,i=1:2,j=1:2],start = R₀[t,i,j])  # n x n matrix start = R₀[t,i,j]

@variable(model,-4.0<= x[i=1:nP]<=4.0,start = x₀[i]) # , start = x₀[i,j]
@variable(model,-4.0<= y[i=1:nP]<=4.0,start = y₀[i]) # , start = y₀[i,j]


@variable(model,-3.0<= ax[i=1:nP,j=1:nP]<=3.0,start = ax₀[i,j]) # , start = ax₀[i,j]
@variable(model,-3.0<= ay[i=1:nP,j=1:nP]<=3.0,start = ay₀[i,j]) # , start = ay₀[i,j]

@variable(model,-100<= b[i=1:nP,j=1:nP]<=100,start = b₀[i,j]) # ,start = b₀[i,j]

# Add the orthogonality constraints: A' * A = I
for i in 1:2
  for j in 1:2
    if i == j
      @NLconstraint(model, [t=1:nP],
       sum(R[t,k,i]*R[t,k,j] for k in 1:2) == 1)  # Diagonal elements
    else
      @NLconstraint(model, [t=1:nP],
      sum(R[t,k,i]*R[t,k,j] for k in 1:2) == 0)  # Off-diagonal elements
    end
  end
end

@NLconstraint(model, [i=1:nP,j=1:nP,k=1:nV;i<j], 
(x[i]+ R[i,1,1]*vt[1,k] + R[i,1,2]*vt[2,k] )*ax[i,j]+ 
(y[i]+ R[i,2,1]*vt[1,k] + R[i,2,2]*vt[2,k] )*ay[i,j]-
b[i,j] <=  0
) 


@NLconstraint(model, [i=1:nP,j=1:nP,k=1:nV;i<j], 
b[i,j] - 
( (x[j]+ R[j,1,1]*vt[1,k] + R[j,1,2]*vt[2,k])*ax[i,j]+ 
(y[j]+ R[j,2,1]*vt[1,k] + R[j,2,2]*vt[2,k] )*ay[i,j])<= 0
)

@NLconstraint(model,[i=1:nP, j=1:nP;i<j], 
    (ax[i,j])^2 + (ay[i,j])^2 ==  1.0       
  ) 

@NLconstraint(model, 
[i=1:nP,k=1:nV], 
(x[i] + R[i,1,1]*vt[1,k] + R[i,1,2]*vt[2,k] )^2 + 
(y[i] + R[i,2,1]*vt[1,k] + R[i,2,2]*vt[2,k] )^2 ≤ rd^2              
)

#println(model)
# Solve the model
set_silent(model)
solTime =@elapsed optimize!(model);
# optimize!(model)
# model,xt,yt,axt,ayt,bt,Rt,vt
 return model,x,y,ax,ay,b,R,vt,solTime 
end 

#for graphics in tikz 

function  drawP(n,x,y,R,vt,nPrint,rd)  
  nV = size(vt[1,:])[1] 
  file = "draw"*string(nPrint)*".txt"
  io = open(file,"w") 
  @printf(io,"\\draw(0,0) circle (%4.2f);\n",rd)  
    for i=1:n    
      @printf(io,"\\draw[fill={{rgb:red,%2.1f;green,%2.1f;blue,%2.1f}}]",
          rand(), rand(),rand())     
      for ii=1:nV #[1,2,4,5]
        xf = value(x[i]) + value(R[i,1,1])*vt[1,ii] + 
                        value(R[i,1,2])*vt[2,ii] 

        yf = value(y[i])+ value(R[i,2,1])*vt[1,ii] + 
                      value(R[i,2,2])*vt[2,ii] 
    
        @printf(io,"(%4.2f,%4.2f)--",xf,yf) 
      end 
    @printf(io,"cycle;\n") 
  end        
  close(io)
end

# =============

function drawJ(n, x, y, R, vt, nPrint, rd)
  fig = Figure(resolution = (600, 600))
  ax = Axis(fig[1,1]; aspect = DataAspect())

  nV = size(vt, 2)
  polys = Vector{Vector{Point2f}}(undef, n)
  cols = [RGBAf(rand(), rand(), rand(), 0.3) for _ in 1:n]

  for i in 1:n
      pts = Vector{Point2f}(undef, nV)
      xi, yi = value(x[i]), value(y[i])
      for k in 1:nV
          xf = xi + value(R[i,1,1])*vt[1,k] + value(R[i,1,2])*vt[2,k]
          yf = yi + value(R[i,2,1])*vt[1,k] + value(R[i,2,2])*vt[2,k]
          pts[k] = Point2f(xf, yf)
      end
      polys[i] = pts
  end

  poly!(ax, polys; color = cols, strokecolor = :black, strokewidth = 0.5)

  θ = range(0, 2π, length = 200)
  lines!(ax, rd*cos.(θ), rd*sin.(θ); color = :black, linewidth = 1)
  limits!(ax, -rd - maximum(abs, vt), rd + maximum(abs, vt),
           -rd - maximum(abs, vt), rd + maximum(abs, vt))

  save("drawing.png",fig)
end

# =============

function den(n,V,r,rd)  
  # density 
  num = n*V*r^2*sin(2.0*π/V)/2.0 
  den = π*rd^2 
  return num/den   
end

function upB(V,r,rd) 
  # upper bound 
  num = π*rd^2 
  den = V*r^2*sin(2.0*π/V)/2.0 
  return num/den 
end

function dataS(file,iter,n,V,r,rd,TrialN,solTime)
  #file =  "dataSol.txt"
  temp0 = den(n,V,r,rd) 
  temp1 = upB(V,r,rd) 
  io = open(file,"w")  
  @printf(io,"%d & %d  & %d  & %4.2f & %8.4f & %4.2f & %d & %12.4f \n",
              iter, n,    V,    r,  temp0,temp1,TrialN,solTime) 
  #@printf(io,"(%d,%4.2f)\n",n,solTime)          
  close(io)
end 

function  packOne(numP,rr,nV,rd)
  file =  "dataSol.txt"
  io = open(file,"w")  
  close(io)

  timeLimit= 60.0  #  50 seconds   10 minutos
  startTime= time()

  solTime  = 0.0   
  TrialP = 0   
  TrialN = 0   

  iterMax = 10
  V = nV
  n  = numP 
  temp = true
  iter = 0   
  ϵ = 0.3 

  xt,yt,axt,ayt,bt,Rt,vt = nothing, nothing, nothing, nothing, nothing, nothing, nothing 

  while  ( time()- startTime) < timeLimit 
    if n == 2 
      x₀ =  -0.5*ϵ .+ ϵ*rand(n)
      y₀ =  -0.5*ϵ .+ ϵ*rand(n)
      
      ax₀ = -0.5*ϵ .+ ϵ*rand(n,n)
      ay₀ = -0.5*ϵ .+ ϵ*rand(n,n)
      b₀ = -0.5*ϵ .+ ϵ*rand(n,n)       
      R₀ = -0.5*ϵ .+ ϵ*rand(n,2,2)
  
      model,xt,yt,axt,ayt,bt,Rt,vt =pkFun(n,x₀,y₀,ax₀,ay₀,b₀,R₀,rr,V,rd)   
      #drawP(n,x₀,y₀,R₀,vt,true,0)  
      #model,x,y,R,vt =pkFun(n,x₀,y₀,ax₀,ay₀,b₀,R₀,rr,V)   
      temp = termination_status(model) == LOCALLY_SOLVED   

      if  temp
        n = n + 1
      end 
      iter = iter + 1 
    elseif  temp 
      x₀ =  ϵ*rand(n,V)
      y₀ =  ϵ*rand(n,V)  
      ax₀ = ϵ*rand(n,n)
      ay₀ = ϵ*rand(n,n)
      b₀ =  ϵ*rand(n,n)       
      R₀ =  ϵ*rand(n,2,2)
  
      x₀[1:n-1] = value.(xt)  
      y₀[1:n-1] = value.(yt)  
      ax₀[1:n-1,1:n-1] =value.(axt) 
      ay₀[1:n-1,1:n-1] =value.(ayt) 
      b₀[1:n-1,1:n-1] = value.(bt) 
      R₀[1:n-1,:,:] = value.(Rt)     
      #if iter ==1
      #drawP(n,x₀,y₀,R₀,vt,true,"Ini")  
      #end       
      model,xt,yt,axt,ayt,bt,Rt,vt,solT=pkFun(n,x₀,y₀,ax₀,ay₀,b₀,R₀,rr,V,rd)
      solTime = solTime + solT   
      temp = termination_status(model) == LOCALLY_SOLVED         
      if temp               
        drawJ(n,xt,yt,Rt,vt,"sol",rd)  
        dataS(file,iter,n,V,rr,rd,TrialN,solTime)
        TrialP = TrialP + 1
        n = n + 1
      end      
    else 
      x₀  = ϵ*value.(xt)  
      y₀  = ϵ*value.(yt)  
      ax₀ = ϵ*value.(axt) 
      ay₀ = ϵ*value.(ayt) 
      b₀  = ϵ*value.(bt) 
      R₀  = ϵ*value.(Rt) 
      model,xt,yt,axt,ayt,bt,Rt,vt,solT=pkFun(n,x₀,y₀,ax₀,ay₀,b₀,R₀,rr,V,rd)
      solTime = solTime + solT   
      #TrialN = TrialN + 1 
      temp = termination_status(model) == LOCALLY_SOLVED           
      if !temp
        TrialN = TrialN + 1
      end 
      if temp
        drawJ(n,xt,yt,Rt,vt,"sol",rd)  
        dataS(file,iter,n,V,rr,rd,TrialN,solTime)                
        n = n + 1
      end 
    end   
    iter = iter + 1 
    @printf("iIter: %d, Itens: %d TrialN: %d time: %4.2f \n",
    iter,n,TrialN,solTime)      
  end 
  @printf("It was Packed %d time: %2.1f\n",n-1,solTime)
  #return model 
end 
# numP,rr,nV,rd
packOne(2,1.0,5,4.0)  
