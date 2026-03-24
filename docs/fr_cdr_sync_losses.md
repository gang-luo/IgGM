# FR/CDR Sync Losses

## Loss modes

Lightning loss config now supports:

- `legacy`
- `fr_cdr_boltz`
- `fr_cdr_iggm`

## Shared supervised terms

Both FR/CDR modes consume `outputs["3d"]["fr_cdr"]` and optimize:

- `loss_fr`
  - FR coordinate MSE
  - FR rotation target loss
  - FR translation target loss
- `loss_cdr_local`
  - loop-local all-atom coordinate MSE
- `loss_occupancy`
  - BCE-with-logits on prefix-valid occupancy targets
- `loss_seam`
  - endpoint local-coordinate consistency for loop ends
- `loss_clash`
  - CA-level short-range clash penalty inside predicted loops

## Style difference

- `fr_cdr_boltz`
  - balanced FR / local / occupancy weighting
- `fr_cdr_iggm`
  - slightly stronger FR emphasis and slightly weaker occupancy emphasis

These are lightweight training objectives meant to unblock FR/CDR finetuning without changing the legacy loss path.
