# Open document
openfemm
opendocument('femm/v_shape.FEM');

# Peak phase current
I_S = 200
beta = 0.6454

# d-q currents
I_d = -I_S*cos(beta)
I_q = I_S*sin(beta)

# Number of pole pairs
p = 2

# Rotate the rotor at various positions and save the file
# such that these can be analyzed using Python
angle_step = 5
for i = 0:angle_step:360
  if (i > 0)
    # Rotate rotor
    mi_selectgroup(1);
    mi_selectgroup(2);
    mi_moverotate(0,0,-angle_step)
  end

  # Change phase currents
  I_a = I_d*cosd(p*i)-I_q*sind(p*i)
  I_b = I_d*cosd(p*i-120)-I_q*sind(p*i-120)
  I_c = I_d*cosd(p*i+120)-I_q*sind(p*i+120)

  mi_modifycircprop('A', 1, I_a)
  mi_modifycircprop('B', 1, I_b)
  mi_modifycircprop('C', 1, I_c)


  # Save file
  filename = sprintf("femm/Rotations/v_shape_%d.FEM", i)
  mi_saveas(filename)
end

