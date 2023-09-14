// static L1: f64 = 0.055;
static L2: f64 = 0.315;
static L3: f64 = 0.045;
static L4: f64 = 0.108;
static L5: f64 = 0.005;
static L6: f64 = 0.034;
static L7: f64 = 0.015;
static L8: f64 = 0.088;
static L9: f64 = 0.204;

fn a_1_to_0(th_1: f64, x: (f64, f64, f64)) -> (f64, f64, f64) {
    return (
        (x.0 + L6) * th_1.cos() - (-x.2 - L4) * th_1.sin(),
        (x.0 + L6) * th_1.sin() + (-x.2 - L4) * th_1.cos(),
        x.1 + L2 + L3,
    );
}

fn a_2_to_1(th_2: f64, x: (f64, f64, f64)) -> (f64, f64, f64) {
    return (
        (x.0 + L7) * th_2.cos() - (x.1 - L8) * th_2.sin(),
        (x.0 + L7) * th_2.sin() + (x.1 - L8) * th_2.cos(),
        x.2 - L5,
    );
}

fn a_3_to_2(th_3: f64, x: (f64, f64, f64)) -> (f64, f64, f64) {
    return (
        x.0 * th_3.cos() - (x.1 - L9) * th_3.sin(),
        x.0 * th_3.sin() + (x.1 - L9) * th_3.cos(),
        x.2,
    );
}

pub fn ee_pos(th_1: f64, th_2: f64, th_3: f64) -> (f64, f64, f64) {
    return a_1_to_0(th_1, a_2_to_1(th_2, a_3_to_2(th_3, (0.0, 0.0, 0.0))));
}
