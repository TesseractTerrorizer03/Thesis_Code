"""Additional protocol and instrumentation checks, independent of fitting gate."""
import unittest,inspect
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import kent_experiment as e


class ProtocolChecks(unittest.TestCase):
    def test_instrumentation_preserves_baselines(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(300,9))*.01
        with threadpool_limits(limits=1):
            initial=e.b.make_initializations(x,1)
            for family in ('GMM','vMF unlinked','vMF linked'):
                spec=next(s for s in e.specs() if s['family']==family and s['k']==1)
                original_family={'GMM':'GMM','vMF unlinked':'Unlinked','vMF linked':'Linked'}[family]
                before=e.b.fit_candidate(x,e.b.Spec(original_family,1,spec['covariance'] if family=='GMM' else '',spec['regularization']),initial)
                after=e.fit_baseline(x,spec,initial)
                np.testing.assert_allclose(e.b.score_model(x,before),e.b.score_model(x,after),atol=1e-12)
                self.assertEqual([r['accepted'] for r in before['diagnostics']],
                                 [r['accepted'] for r in after['diagnostics']])
                for row in after['diagnostics']:
                    self.assertGreater(row['elapsed_seconds'],0)
                    self.assertEqual(len(row['history']),len(row['times']))

    def test_grid_and_selection_only_use_validation(self):
        rows=pd.DataFrame(e.specs());rows['status']='ok'
        rows['validation_mean_loglik']=-abs(rows.k-3)+np.arange(len(rows))*1e-8
        chosen=e.select(rows)
        self.assertEqual(set(rows.k),{1,3,6,10})
        self.assertEqual(len(rows),64)
        for view,group in chosen.groupby('view'):
            if view.startswith('same_K'):self.assertEqual(group.k.nunique(),1)
            if view.startswith('budget10'):self.assertLessEqual(group.parameters.max()/group.parameters.min(),1.1)
            if view.startswith('exact'):self.assertEqual(group.parameters.nunique(),1)
        self.assertNotIn('test',inspect.signature(e.select).parameters)
        for _,group in chosen[chosen.view.str.startswith('budget10')].groupby('view'):
            self.assertFalse({'Kent unlinked','vMF unlinked'}.issubset(set(group.family)))


if __name__=='__main__':unittest.main(verbosity=2)
