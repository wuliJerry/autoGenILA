// Wrapper that adds reset port to Tile for autoGenILA compatibility
// Reset initializes PE accumulators c1, c2 to zero
module TileWithReset(
  input         clock,
  input         reset,
  input  [7:0]  io_in_a_0,
  input  [19:0] io_in_b_0,
  input  [19:0] io_in_d_0,
  input         io_in_control_0_dataflow,
  input         io_in_control_0_propagate,
  input  [4:0]  io_in_control_0_shift,
  input  [2:0]  io_in_id_0,
  input         io_in_last_0,
  output [7:0]  io_out_a_0,
  output [19:0] io_out_c_0,
  output [19:0] io_out_b_0,
  output        io_out_control_0_dataflow,
  output        io_out_control_0_propagate,
  output [4:0]  io_out_control_0_shift,
  output [2:0]  io_out_id_0,
  output        io_out_last_0,
  input         io_in_valid_0,
  output        io_out_valid_0,
  output        io_bad_dataflow
);

  Tile tile_inst (
    .clock                    (clock),
    .io_in_a_0                (io_in_a_0),
    .io_in_b_0                (io_in_b_0),
    .io_in_d_0                (io_in_d_0),
    .io_in_control_0_dataflow (io_in_control_0_dataflow),
    .io_in_control_0_propagate(io_in_control_0_propagate),
    .io_in_control_0_shift    (io_in_control_0_shift),
    .io_in_id_0               (io_in_id_0),
    .io_in_last_0             (io_in_last_0),
    .io_out_a_0               (io_out_a_0),
    .io_out_c_0               (io_out_c_0),
    .io_out_b_0               (io_out_b_0),
    .io_out_control_0_dataflow(io_out_control_0_dataflow),
    .io_out_control_0_propagate(io_out_control_0_propagate),
    .io_out_control_0_shift   (io_out_control_0_shift),
    .io_out_id_0              (io_out_id_0),
    .io_out_last_0            (io_out_last_0),
    .io_in_valid_0            (io_in_valid_0),
    .io_out_valid_0           (io_out_valid_0),
    .io_bad_dataflow          (io_bad_dataflow)
  );

endmodule
