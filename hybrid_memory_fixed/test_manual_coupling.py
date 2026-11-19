"""Mode migration and user-specified coupling-switch timing."""
import unittest
from unittest.mock import patch

import backend as b
from hybrid_settings import validate
from hybrid_panel import panel_value
from assistant_search import locked_controls
from pulse_preview import control_preview


class ManualCouplingTests(unittest.TestCase):
    def test_nominal_preview_shows_each_manual_switch(self):
        settings = validate({'CONSTANT_FLUX':1,'COUPLING_SWITCH_MODE':'manual',
                             'WRITE_COUPLING_START_NS':500.,'READ_COUPLING_START_NS':84.})
        write = control_preview(settings,'write')
        read = control_preview(settings,'read')
        self.assertAlmostEqual(write['coupling_switch_us'],.5)
        self.assertGreater(write['finish_us'],.5)
        self.assertAlmostEqual(read['coupling_switch_us'],.084)
    def test_legacy_settings_migrate_to_explicit_manual_or_always_on(self):
        self.assertEqual(validate({'SWITCH_DESTINATION_COUPLING':0})['COUPLING_SWITCH_MODE'],'always_on')
        with self.assertWarns(UserWarning):
            self.assertEqual(validate({'SWITCH_DESTINATION_COUPLING':1})['COUPLING_SWITCH_MODE'],'manual')
        with self.assertWarns(UserWarning):
            self.assertEqual(validate({'COUPLING_SWITCH_MODE':'automatic'})['COUPLING_SWITCH_MODE'],'manual')
        self.assertEqual(validate({'COUPLING_SWITCH_MODE':'Always on'})['COUPLING_SWITCH_MODE'],'always_on')
        self.assertEqual(panel_value('COUPLING_SWITCH_MODE','always_on'),'always_on')
        for value in (-1, 2, 'bad'):
            with self.assertRaises(ValueError):
                validate({'SWITCH_DESTINATION_COUPLING':value})
        for key in ('WRITE_COUPLING_START_NS','READ_COUPLING_START_NS'):
            with self.assertRaises(ValueError):
                validate({key:-1})

    def test_manual_values_are_direction_specific_and_do_not_run_pilot(self):
        with patch.dict(b._SETTINGS, {'CONSTANT_FLUX':1,'CONSTANT_FIELD':0,
                        'COUPLING_SWITCH_MODE':'manual','WRITE_COUPLING_START_NS':37.5,
                        'READ_COUPLING_START_NS':84.0}), patch.object(b,'cavity_gate_time') as pilot:
            write = b.selected_coupling_switch('write')
            read = b.selected_coupling_switch('read')
            self.assertAlmostEqual(write['time_us'],.0375)
            self.assertAlmostEqual(read['time_us'],.084)
            self.assertEqual(write['mode'],'manual')
            self.assertIsNone(write['mean_photons'])
            pilot.assert_not_called()
            locks = locked_controls(b._SETTINGS)
            self.assertEqual(locks['COUPLING_SWITCH_MODE'],'manual')
            self.assertEqual(locks['WRITE_COUPLING_START_NS'],37.5)
            with patch.dict(b._SETTINGS, {'COUPLING_SWITCH_MODE':'always_on'}):
                self.assertIsNone(b.selected_coupling_switch('write'))
            with patch.dict(b._SETTINGS, {'CONSTANT_FLUX':0,'CONSTANT_FIELD':0}):
                self.assertIsNone(b.selected_coupling_switch('write'))

    def test_late_manual_read_switch_still_searches_complete_retrieval(self):
        with patch.dict(b._SETTINGS, {'CONSTANT_FLUX':1,'CONSTANT_FIELD':0,
                        'COUPLING_SWITCH_MODE':'manual','READ_COUPLING_START_NS':500.}), \
             patch.object(b,'PHI_PARK',b.PHI_SWAP), patch.object(b,'STORAGE',0.), \
             patch.object(b,'_READ_GATE_TIME',.5), patch.object(b,'_READ_CLEANUP_START',None), \
             patch.object(b,'_READ_TC_START',None):
            result = b.optimize_constant_flux_readout(samples=2)
            self.assertEqual(result['samples'],2)
            self.assertGreaterEqual(result['cleanup_start_us'],.5+b.GAP)
            b._READ_CLEANUP_START = result['cleanup_start_us']
            inputs = ((1,0),(0,1),(1,1),(1,1j))
            finals = [b.evolve_read(b.stored_retrieval_initial(a,c))[1][-1] for a,c in inputs]
            self.assertAlmostEqual(b._six_state_mean(finals,'read'),result['mean_six_state_joint_fidelity'],places=5)


if __name__ == '__main__':
    unittest.main()
