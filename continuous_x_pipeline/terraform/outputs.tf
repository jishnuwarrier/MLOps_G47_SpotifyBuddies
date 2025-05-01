output "floating_ip_out" {
  description = "Floating IP assigned to node1"
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

#-- newly added --
output "data_volume_id" {
  description = "Volume ID for persistent data"
  value       = openstack_blockstorage_volume_v3.data_volume.id
}